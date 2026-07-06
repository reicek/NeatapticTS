# NEAT Genesis EvoDevo: Racing Curriculum Log

**Status:** [DONE]

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

### Phase 4 — Tier 3: 2v2 no pits

- [DONE] Step 01 — Plan Tier 3 boundary: recorded team layout `[0, 0, 1, 1]`, 91-dim observation (70 base + 21 teammate-radio), role-divergence seam, shared-equal team-fitness (DR-001), NGE primitive risk assessment; authored Step 02-07 packets.
- [DONE] Step 02 — Research 2v2 coevolution and role-divergence contracts: documented 91-dim observation, 4-distinct-genome coevolution scaling, `createTeamFitnessEvaluator` shared-equal compatibility, worker-side adaptation feasibility (DR-002); confirmed `planGrowthMorphs` and `computeFocusScores` are externally configurable (DR-003).
- [DONE] Step 03 — Red tests for two-car team runtime: 11 red tests across `observation.assembler.test.ts`, `simulation-worker.coevolution.test.ts`, `simulation-worker.race-pack.test.ts`, and `browser-entry.test.ts`; all fail for the right reasons.
- [DONE] Step 04 — Implement 2v2 worker evaluation loop: 4 implementation slices — teammate observation + four-genome coevolution, shared-equal team fitness, 4-car browser rendering with per-car controllers, worker-side continuous adaptation relocation (DR-002/05).
- [DONE] Step 05 — Green validation and regression triage: focused Jest slices passed; Chrome DevTools MCP visual validation confirmed 4-car Tier 3 simulation with worker-side adaptation; 3 pre-existing race-pack test failures triaged as carry-forward debt.
- [DONE] Step 06 — Document Tier 3 contract: `examples/racing_curriculum/README.md` updated with Tier 3 2v2 contract; `npm run docs` and `npm run lint` passed.
- [DONE] Step 07 — Logging and tracker handoff: Phase 4 compressed into this log, Phase 4 marked [DONE], Phase 5 advanced to [WIP].

### Phase 5 — Tier 4: 2v2 tires and pits

- [DONE] Step 01 — Plan Tier 4 boundary: recorded tire degradation model (pinned Tier 4 formula, exponential decay, grip multiplier), pit-stop mechanics (4-tick duration, 3 slots per team per DR-004, own-team entry, tire restoration), pit-entrance blocking (emergent from car separation physics), NGE primitive dependency assessment (DR-005); authored Step 02-07 packets.
- [DONE] Step 02 — Research tire/pit mechanics and NGE primitive dependencies: source-grounded research brief confirmed 95-channel Tier 4 observation (91 Tier 3 + 4 own-car tire health), documented tire decay/pit lifecycle NOT wired into worker race-pack (GAP), corrected DR-005 via DR-005-CORRECTION (EpisodicSlot/GatingRouter DO exist as genome-level computation motifs, not episode-level), identified 4 files needing modification.
- [DONE] Step 03 — Red tests for tire/pit contracts: 9 red tests across `simulation-worker.coevolution.test.ts` (2 tests: 95-input genomes) and `simulation-worker.race-pack.test.ts` (7 tests: 95-channel obs, tire in obs tail, tire decay, grip multiplier, pitStatus defined, sentinel init, opposing-team exclusion). All fail for the right reasons.
- [DONE] Step 04 — Implement tire/pit layer: 2 implementation slices — `p5-s04-impl` (observation extension to 95 channels, coevolution wiring with TIER_FOUR_CONTROLLER_INPUT_SIZE=95, tire decay + pit lifecycle wired into worker tick, browser-entry tier 4 options) and `p5-s04-green` (green validation). All 9 previously-red tests now pass. 58 tests pass (2 suites). No src/ files touched.
- [DONE] Step 05 — Green validation and regression triage: 45 suites / 385 tests ALL PASS. tsc (tsconfig.json) clean. Lint 0 issues. Build:racing-curriculum OK (719.8kb). Chrome DevTools MCP visual: Tier 4 simulation running, tire markers visible (22 white pixels), pit overlays visible (69 blue team-A pixels), 0 console errors. Plan validators both PASS.
- [DONE] Step 06 — Document Tier 4 contract: `examples/racing_curriculum/README.md` updated with ~260-line Tier 4 2v2 tires-and-pits contract section (3 Mermaid diagrams: tire decay flowchart, pit lifecycle state diagram, Tier 4 feedback loop), 95-channel observation vector table, tire decay formula, grip multiplier explanation. JSDoc improved on 4 source files. Academic-docs-auditor audit completed. `npm run docs` and `npm run lint` passed.
- [DONE] Step 07 — Logging and tracker handoff: Phase 5 compressed into this log, Phase 5 marked [DONE], Phase 6 advanced to [WIP].

### Phase 6 — Tier 5: 3v3 full

- [DONE] Step 01 — Plan Tier 5 boundary: recorded team layout [0,0,0,1,1,1], 95-channel observation (21 radio already included — Tier 5 fully populates 3 rows), 42-float shared radio slab, tire/pit carry-forward from Tier 4, role-divergence observables to define, polyandric reproduction core primitive available but benchmark wiring missing, coevolution container needs 6-car branch. NGE primitive assessment: reproducePolyandric CONFIRMED, NgeReproductionPolicy CONFIRMED, ModulatorBroadcaster/EpisodicSlot/GatingRouter CONFIRMED at genome level. DR-006 (fitness policy split) and DR-007 (renderer colors blue/red canonical) recorded. Authored Step 02-07 packets.
- [DONE] Step 02 — Research 3v3 full-team contracts: source-grounded research brief with 9 findings (R1-R9). Identified coevolution 6-car branch integration point, polyandric reproduction wiring path with 5 blockers (P1-P5), fitness policy split resolution, 4 role-divergence observable metrics (blockerDelta, lane-hold time, radio MI, within-team variance), full radio population changes (self-broadcast for 3-car teams), 6-car rendering (pit overlay stride fix), 6-car race-pack changes (team layout, pitStatus, tickPitLifecycle), and 8 implementation seams.
- [DONE] Step 03 — Red tests for three-car team runtime: 4 coevolution tests (6-car allocation) + 8 race-pack tier5 tests (2 radio population, 3 polyandric skip contracts, 3 role-divergence). All non-skipped tests fail for right reasons. 3 polyandric tests skipped (P1/P2 blockers).
- [DONE] Step 04 — Implement 3v3 full evaluation loop: TIER_FIVE_CAR_COUNT=6 branch added to coevolution service, self-broadcast for 3-car teams in observation assembler, role-divergence service created (computeRoleDivergenceMetrics with blockerDelta and inferredRole), 6-element pitStatus with layout-aware stride in race-pack and renderer. Polyandric reproduction DEFERRED (P1/P2 blockers). 16 suites, 152 tests pass, 3 skipped. tsc clean, lint 0, build 719.9kb OK.
- [DONE] Step 05 — Green validation and regression triage: 46 suites / 394 tests ALL PASS (0 failures), 3 skipped (polyandric P1/P2). tsc (tsconfig.json) clean. Lint 0 issues. Build 719.9kb OK. Chrome DevTools MCP visual: Tier 5 simulation confirmed (N101/C388, STABLE, 0 console errors). Plan-sync, agent-graph, plan-phase-packets gates all PASS. 27 tsc.test.json errors verified as carry-forward (reduced from 46 by Phase 6 work).
- [DONE] Step 06 — Document Tier 5 contract: README updated with ~385-line Tier 5 contract section (3 Mermaid diagrams, 95-channel observation table clarifying 21 radio already included, 42-float shared radio slab, polyandric reproduction policy, role-divergence observables, 6-car rendering). JSDoc improved on 5 source files (atemporal fixes). npm run docs exit 0, npm run lint exit 0. cortex-index PASS, routing-table-freshness PASS.
- [DONE] Step 07 — Logging and tracker handoff: Phase 6 compressed into this log, Phase 6 marked [DONE], Phase 7 advanced to [WIP].

### Phase 7 — Tier 6: 3v3 advanced strategy

- [DONE] Step 01 — Plan Tier 6 boundary: analytics-only fallback (DR-008), FSM 5-bug fix planned (DR-009), NGE primitive assessment table recorded, Step 02-07 packets authored. plan-sync + step-packet gates PASS.
- [DONE] Step 02 — Research hall-of-fame wiring: 9 findings (R1-R9). OpponentSnapshotPool API mapped. FSM 5 compounding bugs identified. Tire physics FULLY IMPLEMENTED (false positive). Strategy-divergence = NEW module. 27 tsc errors confirmed.
- [DONE] Step 03 — Red tests: 15 red tests across 3 files (7 multi-generation + 3 tier6 HoF/adapter + 5 strategy-divergence). All fail for right reasons. Types from source modules.
- [DONE] Step 04 — Implement Tier 6 evaluation loop: 3 slices (fsm-bugfix, hof-wiring, analytics) + 1 REMOVED (tire-physics). FSM 5 bugs fixed. OpponentSnapshotPool wired with type adapter. Strategy-divergence analytics module created. 68 suites / 502 tests pass.
- [DONE] Step 05 — Green validation: 68 suites / 502 tests ALL PASS (3 skipped polyandric). tsc clean, 27 carry-forward unchanged, lint 0, build 719.9kb OK. plan-sync PASS. No regressions.
- [DONE] Step 06 — Document Tier 6 contract: 4 source files documented, 3 Mermaid diagrams + 3 citations, README regenerated 1258→1739 lines, readiness checklist 6 items marked [x]. tsc clean, lint 0.
- [DONE] Step 07 — Logging and tracker handoff: Phase 7 compressed into this log, Phase 7 marked [DONE]. Carry-forward blockers documented for nge-core-algorithm handoff.

## Phase 6 — Tier 5: 3v3 full — COMPLETED

**Status:** [DONE] — all steps (Step 01 through Step 07) complete. Phase 6 compressed. Polyandric reproduction DEFERRED (P1/P2 blockers — nge-core-algorithm ownership).

### Step summary (Step 01 – Step 07)

- [DONE] Step 01 — Plan Tier 5 boundary: recorded Tier 5 3v3 full-team boundary decisions — team layout `[0,0,0,1,1,1]` (6 cars, 3 per team), 95-channel observation (unchanged from Tier 4 — 21 radio channels already included, Tier 5 fully populates all 3 teammate-radio rows), 42-float shared radio slab (6 cars × 7 channels), tire/pit mechanics carry-forward from Tier 4, role-divergence observables to define (blockerDelta, lane-hold time, radio MI, within-team variance), polyandric reproduction core primitive available but benchmark wiring missing, coevolution container caps at 4 cars (needs TIER_FIVE_CAR_COUNT=6 branch), fitness policy conflict (DR-006 split policy). NGE primitive assessment: reproducePolyandric AVAILABLE, NgeReproductionPolicy (mode='polyandric') AVAILABLE, ModulatorBroadcaster/EpisodicSlot/GatingRouter AVAILABLE at genome level (episode-level deferred to nge-core-algorithm). DR-006 (fitness policy: best-finishing for queen, shared-equal for population) and DR-007 (renderer colors: blue/red canonical) recorded. Authored Step 02-07 packets with red-green slices.
- [DONE] Step 02 — Research 3v3 full-team contracts: source-grounded research brief with 9 findings (R1-R9):
- R1 (Coevolution Container): `createCoevolutionContainer` caps at 4 cars — needs `TIER_FIVE_CAR_COUNT=6` branch. Pure branching logic change, no blocker.
- R2 (Polyandric Reproduction): `reproducePolyandric` available but 5 blockers identified — P1 (NGE_DNA adoption gap: racing uses Network, not NgeDnaCanonicalEnvelope), P2 (NgePolyandricInput/NgePolyandricDroneInput not exported), P3 (racing FSM has no reproduction step), P4 (schema mismatch: reference spec uses non-overlapping/queen-weighted, implemented uses roundRobin/byFitness/bySpecialization), P5 (queenBias not honored by merge logic). Ownership split: nge-core-algorithm owns P1/P2/P4/P5, nge-benchmark-workflow owns P3.
- R3 (Fitness Policy): Split-policy resolution — queen selection uses `selectBestFinishingPosition` (Policy A, coevolution.service.ts), population fitness uses `computeSharedEqualTeamFitness` (Policy B, evolution.protocol.service.ts). No new core aggregation function required.
- R4 (Role-Divergence Metrics): 4 observable metrics defined — (a) blockerDelta (leave-one-out team-score contribution), (b) lane-hold time in opponent pit corridor, (c) radio mutual information, (d) within-team position variance. Queen/blocker/pacer roles emerge from identical DNA — never hardcoded. Observability-only, does not change fitness.
- R5 (Full Radio Population): `buildTeammateRadioSlots` excludes focal car — must include self as one of 3 slots for 3-car teams. Risk: global change affects Tier 3/4. Decision: Tier-5-specific self-inclusion when sameTeamCount >= 3.
- R6 (6-Car Rendering): Structurally supported (TIER_FIVE_TEAM_LAYOUT exists, generic car iteration). Gap: `drawPitOverlays` assumes 4-element pitStatus layout — must use layout-aware stride (teamIndex * 3 + 1 for 6-element).
- R7 (6-Car Race-Pack): Generic tick loop works for 6 cars. Gaps: team layout fallback wrong for 6 cars, pitStatus 4-element needs 6-element, tickPitLifecycle/resolvePitEntries need 3-per-team handling, observation tier selection, radioField slab dead (needs 42-float allocation).
- R8 (Implementation Seams): 8 independently testable seams ordered by priority.
- R9 (Blockers Summary): P1-P5 polyandric blockers with ownership and severity. Cortex RAG gap for examples/ files (LOW — native view fallback works).
- [DONE] Step 03 — Red tests for three-car team runtime: 12 red tests across 2 test files:
- `simulation-worker.coevolution.test.ts` (4 tests): 6-car genome layout — service returns 4 genomes instead of 6, team layout [0,0,1,1] instead of [0,0,0,1,1,1], carIndex range wrong, genome 5 undefined. Root cause: no Tier 5 branch in `createCoevolutionContainer`.
- `simulation-worker.race-pack.tier5.test.ts` (NEW, 8 tests): 2 radio population tests (slot 2 all-zero — self-exclusion in teammate filter), 3 polyandric `it.skip()` contracts (BLOCKED P1/P2 — document expected behavior), 3 role-divergence tests (module not found — Step 04 creates it).
- Result: coevolution 19 passed/4 failed, race-pack 49 passed/5 failed/3 skipped. All new red tests fail for right reasons, no regression.
- [DONE] Step 04 — Implement 3v3 full evaluation loop: 2 implementation slices (`p6-s04-impl`, `p6-s04-green`):
- `p6-s04-impl` [DONE]: Added `TIER_FIVE_CAR_COUNT=6` and `TIER_FIVE_TEAM_LAYOUT=[0,0,0,1,1,1]` to coevolution.service.ts with `isTier5` branch (replaces old isTier3-only ternary, no dual-path). Exported `selectQueenPerTeam` (standalone, observability-only). `buildTeammateRadioSlots` includes self-broadcast for 3-car teams (sameTeamCount >= 3). Created `simulation-worker.role-divergence.service.ts` with `computeRoleDivergenceMetrics` (blockerDelta via leave-one-out, inferredRole classification). 6-element `Int16Array` pitStatus with 3-per-team layout in race-pack service. Layout-aware pit-status stride in renderer (`teamIndex * 3 + 1` for 6-element). 42-float radioField allocation. Polyandric reproduction call DEFERRED — P1/P2 blockers remain. 3 skipped polyandric tests remain skipped. Changed files: coevolution.service.ts, observation.assembler.ts, race-pack.service.ts, role-divergence.service.ts (NEW), racing.renderer.ts. tsc OK, lint 0, build 719.9kb OK. jest: 16 suites, 152 passed, 3 skipped.
- `p6-s04-green` [DONE]: Green validation — all Step 03 red tests pass (except 3 skipped polyandric). tsc clean, lint 0, build OK. coverage-guard N/A (no src/ files touched).
- [DONE] Step 05 — Green validation and regression triage: 46 suites / 394 tests ALL PASS (0 failures), 3 skipped (polyandric P1/P2). tsc (tsconfig.json) exit 0. tsc (tsconfig.test.json) 27 pre-existing duplicate identifier errors in 3 files (carry-forward, verified via git stash — reduced from 46 by Phase 6 work). Lint 0 issues. Build:racing-curriculum exit 0 (719.9kb). Chrome DevTools MCP visual: Tier 5 simulation confirmed ("Tier 5 progression step with live NGE inference"), canvas 1600x900 with blue/red team cars, yellow tire/pit markers, network panel N101/C388, status STABLE, 0 console errors. plan-sync PASS, agent-graph PASS (65 agents, 0 issues), plan-phase-packets PASS (0 errors, 0 warnings — fixed Step 03 status mismatch [WIP]→[DONE]). No src/ files changed — coverage-guard N/A.
- [DONE] Step 06 — Document Tier 5 contract: Added ~385-line Tier 5 contract section to `examples/racing_curriculum/README.md` with 3 Mermaid diagrams (Tier 5 feedback loop, polyandric reproduction flow, role-divergence emergence), 95-channel observation vector table (clarifying 21 radio already included — NOT 116), 42-float shared radio slab documentation, polyandric reproduction policy (honestly marked as deferred with no invented timelines), role-divergence observables (blockerDelta, inferredRole), 6-car rendering. Improved JSDoc on 5 source files with atemporal fixes (removed DR-2026 references and P1/P2 labels from coevolution.service.ts, removed DR-2026 reference from role-divergence.service.ts, removed Phase 3 leak from observation.assembler.ts, updated drawPitOverlays JSDoc for 6-element pitStatus). Polyandry Wikipedia citation added. `npm run docs` exit 0, `npm run lint` exit 0, `npx tsc --noEmit` exit 0, `npm run build:racing-curriculum` exit 0. Gates: cortex-index PASS (rebuilt, 1462 docs), routing-table-freshness PASS.
- [DONE] Step 07 — Logging and tracker handoff: Phase 6 compressed into this log, Phase 6 marked [DONE], Phase 7 advanced to [WIP].

### Decision Records (Phase 6)

- **DR-006:** Fitness policy for Tier 5 queen selection. Context: `evolution.protocol.service.ts` uses shared-equal (average) team fitness while `coevolution.service.ts` uses best-finishing-position (min). The reference design says "team wins if any member wins." Chosen: split policy — best-finishing drives queen selection for polyandric reproduction (aligns with reference "queen = best-finishing car"), shared-equal remains for population-level fitness evaluation (preserves Tier 3/4 decision DR-001). Rationale: the two policies serve different purposes — queen selection rewards the winning car's DNA, population fitness rewards team coordination. Rollback: switch to best-finishing for both if split policy causes evolutionary instability. Owner: 01-planning.
- **DR-007:** Renderer team colors. Context: task descriptions reference "cyan (Team A) and magenta (Team B)" but actual renderer code uses blue (#0000ff) and red (#ff0000). Decision: blue/red are canonical (match the implemented code). No color change needed for Tier 5. Rollback: update renderer colors if the reference design's cyan/red-orange spec is later prioritized. Owner: 01-planning.

### Phase 6 changed file groups

- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts` — 6-car coevolution, queen selection
- `examples/racing_curriculum/controller/observation.assembler.ts` — self-broadcast for 3-car teams
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts` — 6-element pitStatus, 42-float radioField, layout-aware stride
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.role-divergence.service.ts` (NEW) — blockerDelta, inferredRole
- `examples/racing_curriculum/renderer/racing.renderer.ts` — layout-aware pit-status stride
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.test.ts` — 4 Tier 5 red tests
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.tier5.test.ts` (NEW) — 8 tests (5 active + 3 skipped)
- `examples/racing_curriculum/README.md` — +385 lines Tier 5 contract section (3 Mermaid diagrams)
- `docs/assets/racing-curriculum.bundle.js` (rebuilt)

### Phase 6 residual risks (carry-forward)

- **Polyandric reproduction DEFERRED (P1/P2 blockers):** P1 — NGE_DNA adoption gap (racing harness uses classic NEAT Network, not NgeDnaCanonicalEnvelope; polyandric operator only accepts/returns NgeDnaCanonicalEnvelope). P2 — NgePolyandricInput/NgePolyandricDroneInput not exported from reproduction.ts. Both owned by nge-core-algorithm. 3 skipped polyandric tests in `simulation-worker.race-pack.tier5.test.ts` (lines 179, 188, 197) remain skipped until P1/P2 resolved.
- **P3 — Racing FSM has no reproduction step:** `handleRaceStep` returns `done: true` but does not transition to `generation-ready` or call any reproduction operator. `advanceTeamGeneration` only increments generation counter. The entire evolutionary loop is a placeholder. Owned by nge-benchmark-workflow.
- **P4 — Schema mismatch with reference.plans.md:** Reference spec specifies `assignedRegionStrategy: "non-overlapping"` and `seedPolicy: "queen-weighted"` — neither value exists in the implemented schema (roundRobin | byFitness | bySpecialization). Core-algorithm decision needed.
- **P5 — queenBias not honored:** `patchPolyandricRegion` merge always lets queen win regardless of `queenBias` value. Works at 1.0 but misleading. Core-algorithm decision.
- **27 tsc.test.json carry-forward errors:** Duplicate identifier errors in `coevolution.test.ts`, `evolution.protocol.test.ts`, `independent-genomes.test.ts` — pre-existing debt, NOT caused by Phase 6 (reduced from 46 by Phase 6 work). Verified via git stash.
- **EpisodicSlot/GatingRouter episode-level composition:** Genome-level primitives exist but have not been evaluated for episode-level pit-timing memory. Escalation to nge-core-algorithm deferred.
- **`examples/` folder coverage:** Outside `collectCoverageFrom` glob — coverage-guard not applicable. All Phase 6 changes were under `examples/`.

**Next boundary:** Phase 7 — Tier 6: 3v3 advanced strategy [WIP]. Continue in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`.

## Phase 5 — Tier 4: 2v2 tires and pits — COMPLETED

**Status:** [DONE] — all steps (Step 01 through Step 07) complete. Phase 5 compressed.

### Step summary (Step 01 – Step 07)

- [DONE] Step 01 — Plan Tier 4 boundary: recorded Tier 4 2v2 tires-and-pits boundary decisions — tire degradation model (4 tires per car [FL, FR, RL, RR], pinned Tier 4 decay formula `baseDecay = |lateralForce|*0.00012 + |longitudinalForce|*0.00006 + |speed|*0.000006`, exponential degradation `wear *= 1 + (1 - tireHealth) * 0.5`, grip multiplier `sqrt(meanTireHealth)`), pit-stop mechanics (4-tick duration `PIT_STOP_TICKS = 4`, own-team `entranceCorridor` AABB entry, roster-order slot claiming, tire restoration to [1,1,1,1] on release), pit-entrance blocking (emergent from `separateCars` collision physics, no additional logic needed), pit slot count (DR-004: keep 3 slots per team, 6 total), NGE primitive dependency assessment (DR-005: EpisodicSlot/GatingRouter initially assessed as NOT available, environment mechanics proceed without them). Authored Step 02-07 packets.
- [DONE] Step 02 — Research tire/pit mechanics and NGE primitive dependencies: source-grounded research brief with 6 findings:
- Finding 1 (GAP): `decayTireState` and grip multiplier NOT wired into worker race-pack — worker `tick()` implements its own simplified centerline-following physics with 5-channel input, not the 91/95-channel observation assembler.
- Finding 2 (CONFIRMED): Exact Tier 4 observation = 95 channels (91 Tier 3 + 4 own-car tire health `[FL, FR, RL, RR]`), NOT 111. Pit status NOT in observation vector. Other cars' tire states NOT in observation. Extension is own-car-only.
- Finding 3 (GAP): `resolvePitEntries` and `tickPitOccupancy` NOT called in worker race-pack — `pitStatus` field exists in Tier 4 race pack but is never updated during evaluation.
- Finding 4 (CONFIRMED): `RuntimeAdaptationEngine` does NOT need tire/pit awareness for basic operation — uses `progress01` as score, mutates topology/weights. Tire/pit awareness is optional enhancement.
- Finding 5 (CONFLICT RESOLVED): EpisodicSlot and GatingRouter DO exist in NGE core as genome-level computation motifs (13 src files). DR-005 contained a factual error caused by flawed `-SimpleMatch` search with pipe character. Corrected by DR-005-CORRECTION: primitives exist at genome level, not episode level; nge-core-algorithm must evaluate genome-level composition for episode-level pit-timing memory.
- Finding 6 (IDENTIFIED): 4 files require changes (simulation-worker.race-pack.service.ts, simulation-worker.coevolution.service.ts, observation.assembler.ts, environment.step.service.ts), 2 optional enhancements (runtime.adaptation.ts, simulation-worker.tier4.ts).
- [DONE] Step 03 — Red tests for tire/pit contracts: 9 red tests across 2 test files:
- `simulation-worker.coevolution.test.ts` (2 tests): Tier 4 genome input size should be 95 not 91 (Expected: 95, Received: 91); all 4 genomes should use 95 inputs (Expected: [95,95,95,95], Received: [91,91,91,91]).
- `simulation-worker.race-pack.test.ts` (7 tests): 95-channel obs (Expected: 95, Received: 5), tire in obs tail (Expected: [0.8,0.7,0.6,0.5], Received: [undefined×4]), tire decay (Expected: <1, Received: 1), grip multiplier (Expected: >0.10131, Received: 0.10131), pitStatus defined (Expected: defined, Received: undefined), sentinel init (Expected: true, Received: false), opposing-team exclusion (Expected: 255, Received: undefined).
- Result: 2 failed/13 passed (coevolution), 7 failed/36 passed (race-pack). All new red tests fail for right reasons, no regression.
- [DONE] Step 04 — Implement tire/pit layer: 2 implementation slices executed:
- `p5-s04-impl` [DONE]: Wired tire decay (`decayTireState`), pit lifecycle (`tickPitOccupancy`, `resolvePitEntries`), and grip multiplier (`sqrt(resolveMeanTireHealth)`) into worker race-pack `tick()` loop. Replaced 5-channel centerline-following physics with 95-channel `assembleTier4Observation` output. Added `TIER_FOUR_CONTROLLER_INPUT_SIZE = 95` to coevolution service. Added tier 4 browser-entry options. No dual-path code — old 5-channel observation fully replaced. Changed files: `simulation-worker.coevolution.service.ts`, `simulation-worker.race-pack.service.ts`, `simulation-worker.race-pack.test.ts`. tsc OK, lint 0 issues, build:racing-curriculum OK. jest: 58 passed, 0 failed (2 suites). All 9 previously-red Tier 4 tests now pass.
- `p5-s04-green` [DONE]: Green validation and coverage guard. jest coevolution 19/19, race-pack 49/49, environment 39/39, observation 32/32 — all pass. tsc (tsconfig.json) PASS. lint PASS. build PASS (719.8kb). plan-sync PASS. plan-phase-packets PASS. coverage-guard N/A (no src/ files touched). No dual-path code confirmed.
- [DONE] Step 05 — Green validation and regression triage: 45 suites / 385 tests ALL PASS (0 failures). tsc (tsconfig.json) exit 0. tsc (tsconfig.test.json) 28 pre-existing duplicate identifier errors in 3 files (carry-forward, verified via git stash). Lint 0 issues. Build:racing-curriculum exit 0 (719.8kb bundle + 5.6mb sourcemap). Plan-sync PASS (0 errors, 0 warnings). Plan-phase-packets PASS (0 errors, 0 warnings). Chrome DevTools MCP visual: Tier 4 simulation running (tick advancing 2365→8833), tire markers visible (22 white=full health pixels), pit overlays visible (69 blue team-A pixels), 0 console errors. No src/ files changed in Phase 5 — coverage-guard N/A.
- [DONE] Step 06 — Document Tier 4 contract: Added ~260-line "Tier 4 2v2 tires-and-pits contract" section to `examples/racing_curriculum/README.md` with 3 Mermaid diagrams (tire decay flowchart, pit lifecycle state diagram, Tier 4 feedback loop), 95-channel observation vector table, tire decay formula, grip multiplier explanation, pit status representation (255 sentinel for no-pit), and code example. Improved JSDoc on 4 source files (`createCoevolutionContainer`, `tickPitLifecycle`, `resolvePitEntries`, `resolvePerCarObservation`, `RacingRenderFrame`). Academic-docs-auditor audit completed: fixed "67 meters" → "7.2 world units", fixed `resolvePitEntries` variable name references, fixed tire decay diagram labels. `npm run docs` exit 0, `npm run lint` exit 0. routing-table-freshness gate PASS. cortex-index gate PASS after rebuild.
- [DONE] Step 07 — Logging and tracker handoff: Phase 5 compressed into this log, Phase 5 marked [DONE], Phase 6 advanced to [WIP].

### Decision Records (Phase 5)

- **DR-004:** Pit slot count per team — keep 3 slots per team (6 total). Reference design says "one pit per team" but TrackPitBox README says "Tier 4+ may generate multiple pit descriptors per team". 6-slot infrastructure was built and tested during Phase 4. Strategic tension at Tier 4 comes from tire degradation + 4-tick pit stop time cost, not from slot scarcity. Rollback: parameterize `PIT_SLOTS_PER_TEAM` by tier if slot scarcity is needed later.
- **DR-005:** NGE primitive dependency — initially assessed EpisodicSlot/GatingRouter as NOT available based on flawed `-SimpleMatch` search. Decision (optB) to proceed with environment mechanics without wiring NGE primitives remains valid.
- **DR-005-CORRECTION:** Supersedes DR-005. EpisodicSlot (`src/neat/genome/genome.types.ts:26`, `genome.utils.ts:88`) and GatingRouter (`genome.types.ts:28`, `genome.utils.ts:138`) DO exist as genome-level computation motifs with typed-array slot storage, materialization factories, and budget enforcement. They exist at genome level, not episode level. Revised escalation: nge-core-algorithm must evaluate whether genome-level primitives can be composed into episode-level pit-timing memory, or whether a new higher-level primitive is needed. Environment mechanics decision (optB) remains valid — proceed without wiring EpisodicSlot/GatingRouter.

### Phase 5 changed file groups

- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.test.ts`
- `examples/racing_curriculum/browser-entry/browser-entry.ts`, `browser-entry.test.ts`
- `examples/racing_curriculum/controller/observation.assembler.ts`
- `examples/racing_curriculum/README.md`
- `docs/assets/racing-curriculum.bundle.js` (rebuilt)

### Phase 5 residual risks (carry-forward)

- **EpisodicSlot/GatingRouter episode-level composition:** Genome-level primitives exist but have not been evaluated for episode-level pit-timing memory. Escalation to `nge-core-algorithm` is deferred — environment mechanics work without them. Phase 6+ may need this evaluation if pit-strategy-aware memory becomes a requirement.
- **28 pre-existing tsc.test.json errors:** Duplicate identifier errors in `coevolution.test.ts`, `evolution.protocol.test.ts`, `independent-genomes.test.ts` — carry-forward debt, NOT caused by Phase 5 work. Verified via git stash on HEAD.
- **`examples/` folder coverage:** Outside `collectCoverageFrom` glob — coverage-guard not applicable. All Phase 5 changes were under `examples/`.
- **Uncommitted Phase 5 implementation changes in working tree** — should be committed separately (not part of logging step scope).

**Next boundary:** Phase 6 — Tier 5: 3v3 full [WIP]. Continue in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`.

## Phase 4 — Tier 3: 2v2 no pits — COMPLETED

**Status:** [DONE] — all steps (Step 01 through Step 07) complete. Phase 4 compressed.

### Step summary (Step 01 – Step 07)

- [DONE] Step 01 — Plan Tier 3 boundary: recorded Tier 3 2v2 boundary decisions — team layout `[0, 0, 1, 1]`, 91-dim observation (70 base + 21 teammate-radio channels), role-divergence seam (genetic divergence + position-dependent observation + team radio), shared-equal team-fitness (DR-001, user-approved), NGE primitive risk assessment (ModulatorBroadcaster/EpisodicSlot/GatingRouter not needed for Tier 3). DR-003 (composite growth pressure) and DR-002 (worker adaptation relocation) and DR-010 (all-cars-coevolve-independently) recorded. Authored Step 02-07 packets.
- [DONE] Step 02 — Research 2v2 coevolution and role-divergence contracts: source-grounded research brief confirmed 91-dim observation (70 base + 3×7 teammate-radio slots), 4-distinct-genome coevolution scaling (`CAR_COUNT` 2→4, team layout `[0,0,1,1]`), `createTeamFitnessEvaluator` shared-equal compatibility (policy-injected, no structural change needed), role-divergence seam (genetic divergence + position-dependent observation + team radio — no hardcoded roles), NGE primitives not needed for Tier 3, `planGrowthMorphs` accepts per-tier growth-budget override, `computeFocusScores` weights externally configurable, worker race-pack runner can host continuous adaptation.
- [DONE] Step 03 — Red tests for two-car team runtime: 11 red tests across 4 test files — `observation.assembler.test.ts` (2 tests: teammateRadioSlots[0] undefined for car 0 and car 2), `simulation-worker.coevolution.test.ts` (4 tests: genomes.length=2 not 4, teamLayout, distinctness, carIndices), `simulation-worker.race-pack.test.ts` (3 tests: resolveTeamFitness undefined, team layout), `browser-entry.test.ts` (2 tests: 2 controllers not 4, networks not all distinct). All fail for the right reasons (missing implementation).
- [DONE] Step 04 — Implement 2v2 worker evaluation loop: 4 implementation slices executed:
- `p4-s04-impl-teammate-obs` [DONE]: Extended `derivePerCarObservationState` to populate teammate radio slots with 7-channel teammate state (position x/y, heading sin, speed, relative offset x/y, relative heading sin). Replaced hardcoded `CAR_COUNT=2`, `CONTROLLER_INPUT_SIZE=4`, `CONTROLLER_OUTPUT_SIZE=2` with tier-dependent constants. Old binary team assignment removed. Tests: observation.assembler 15/15, observation.assembler.tier3 5/5, coevolution 13/13, independent-genomes 16/16. Build 744.9kb.
- `p4-s04-impl-team-fitness` [DONE]: Shared-equal team-fitness aggregation in race-pack service. (Detailed evidence compressed — see PlanUpdate blocks in prior plan version.)
- `p4-s04-impl-browser-4car` [DONE]: `start()` accepts tier options (`{ tier: 3 }`), `DEFAULT_CURRICULUM_TIER=1` fallback, `resolveCurriculumTierFromOptions` helper, `createInitialCurriculumProgress` accepts tier param. Browser-entry tests 77/77 (4 suites), renderer cleanup 16/16. Build OK.
- `p4-s04-impl-worker-adaptation` [DONE]: Relocated `RuntimeAdaptationEngine` from browser main thread into simulation worker. Each car's network evolves continuously via `adaptOnTick` inside the worker's race episode runner `tick()` loop. `CarGenome.getNetwork()` added. `RaceAdaptationContext` type added. `serializeVisualizationNetwork()` serializes car 0's network for browser visualization. Evolution protocol FSM: init stores config, request-generation creates container + per-car adaptation engines, start-race creates runner with adaptation context. Browser-side: removed all `RuntimeAdaptationEngine` imports, constants, types, state, keyboard tuning handler, main-loop adaptation calls, and 12 adaptation helper functions. No dual-path code. Tests: browser-entry 6 suites/76 tests, simulation-worker 14 suites/131 tests. Build 720.8kb.
- [DONE] Step 05 — Green validation and regression triage: tsc (tsconfig.json) OK, lint 0 issues, prettier all 5 changed files pass, build:racing-curriculum OK (720.8kb bundle), browser-entry tests 6/6 suites 76/76 tests PASSED, simulation-worker tests 14/15 suites 131/134 tests passed. Chrome DevTools MCP visual validation at Tier 3: PASS — simulation runs (tick incrementing), network panel updates (N76/C288 → N97/C372 via worker-side adaptation), "LAST CHANGE: WORKER-SIDE ADAPTATION" confirmed, no console errors. 3 pre-existing race-pack test failures triaged (resolveTeamFitness + team layout — from prior unimplemented slices, not caused by worker-adaptation slice).
- [DONE] Step 06 — Document Tier 3 contract: `examples/racing_curriculum/README.md` updated with Tier 3 2v2 contract (team layout, per-car observation with teammate awareness, shared-equal team fitness, role-divergence seam, independent-agent contract). `npm run docs` and `npm run lint` passed.
- [DONE] Step 07 — Logging and tracker handoff: Phase 4 compressed into this log, Phase 4 marked [DONE], Phase 5 advanced to [WIP].

### Decision Records (Phase 4)

- **DR-001:** Team-fitness semantics — shared-equal chosen (user-approved). Both teammates receive the same team-scoped fitness signal.
- **DR-003:** Growth pressure — composite B+C+A approach (lifecycle morph retune + capacity-gated progression + performance-gated complexity bonus). D deferred.
- **DR-002:** Worker adaptation relocation — `RuntimeAdaptationEngine` moved from host main thread into worker. Worker owns continuous adaptation internally.
- **DR-010:** Worker evaluation path — ALL 4 cars coevolve independently. Only blue team #1 network copied to browser for visualization.

### Phase 4 changed file groups

- `examples/racing_curriculum/controller/observation.assembler.ts`, `observation.assembler.test.ts`, `observation.assembler.tier3.test.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts`, `simulation-worker.coevolution.test.ts`, `simulation-worker.independent-genomes.test.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`, `simulation-worker.race-pack.test.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.evolution.types.ts`, `simulation-worker.evolution.protocol.service.ts`
- `examples/racing_curriculum/browser-entry/browser-entry.ts`, `browser-entry.test.ts`
- `examples/racing_curriculum/environment/environment.step.service.ts`
- `examples/racing_curriculum/README.md`
- `src/neat/neat.nge-lifecycle.ts`, `src/neat/nge-juvenile/*` (NGE core growth engine wiring)
- `docs/assets/racing-curriculum.bundle.js` (rebuilt)

### Phase 4 residual risks (carry-forward)

- **3 pre-existing race-pack test failures** (`resolveTeamFitness` + team layout `[0,0,1,1]`): triaged as pre-existing from prior unimplemented slices. The `resolveTeamFitness` function and `[0,0,1,1]` team layout may need verification in Phase 5 or a follow-up fix slice.
- `examples/` folder coverage is outside `collectCoverageFrom` glob — coverage-guard not applicable.
- Uncommitted Phase 4 implementation changes in working tree — should be committed separately (not part of logging step scope).

**Next boundary:** Phase 5 — Tier 4: 2v2 tires and pits [WIP]. Continue in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`.

## Phase 3 — Tier 2: 1v1 with radio — COMPLETED

**Status:** [DONE] — all steps (Step 08 through Step 19) complete; Phase 3 green gate passes (348 tests, lint clean, tsc clean). Phase 3 compressed.

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

| #   | slice_id                                  | title                                                            | goal          | key evidence                                                                                                                                                                              |
| --- | ----------------------------------------- | ---------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `p3-s19-red-obs-team-offset`              | Red tests for team-aware observation offset                      | red-testing   | 7/8 tests, 1 expected failure (teamIndex 1 outer-lane centerline)                                                                                                                         |
| 2   | `p3-s19-impl-obs-team-offset`             | Implement team-aware observation offset                          | implementing  | 8/8 tests; tsc PASS; lint PASS                                                                                                                                                            |
| 3   | `p3-s19-green-obs-team-offset`            | Green validation for team-aware observation offset               | green-testing | 8/8 tests; browser-entry 69/69 across 6 suites; tsc PASS; build:racing-curriculum PASS                                                                                                    |
| 4   | `p3-s19-red-per-car-observation`          | Red tests for per-car observation-state helper                   | red-testing   | 8 passed, 4 failed (derivePerCarObservationState undefined) — honest missing-implementation gap                                                                                           |
| 5   | `p3-s19-impl-per-car-observation`         | Implement per-car observation-state helper                       | implementing  | 12/12 tests; tsc PASS; lint PASS                                                                                                                                                          |
| 6   | `p3-s19-green-per-car-observation`        | Green validation for per-car observation-state helper            | green-testing | 12/12 tests; browser-entry 69/69; tsc PASS; build PASS                                                                                                                                    |
| 7   | `p3-s19-red-browser-per-car-controller`   | Red tests for independent per-car controllers in browser harness | red-testing   | 4 new per-car controller contracts fail (resolveControlFanOut still present, single controller); 70/74 pass                                                                               |
| 8   | `p3-s19-impl-browser-per-car-controller`  | Implement per-car controller Map in browser harness              | implementing  | Map<carIndex, NgeController> maintained; resolveControlFanOut removed; per-car observation wired                                                                                          |
| 9   | `p3-s19-green-browser-per-car-controller` | Green validation for independent per-car browser control         | green-testing | Red tests now pass; no regressions in browser-entry suite                                                                                                                                 |
| 10  | `p3-s19-red-separation-grid`              | Red tests for car separation and worker grid                     | red-testing   | Environment + worker tests fail on overlapping bounding boxes and identical starting positions                                                                                            |
| 11  | `p3-s19-impl-separation-grid`             | Implement car separation and worker starting-grid alignment      | implementing  | CAR_MIN_CENTER_SEPARATION prevents overlap; worker buildRaceFrame staggers cars with grid-spacing                                                                                         |
| 12  | `p3-s19-green-separation-grid`            | Green validation for car separation and worker grid              | green-testing | Red tests now pass; no regressions; baselines updated                                                                                                                                     |
| 13  | `p3-s19-red-renderer-cleanup`             | Red tests for renderer visual cleanup                            | red-testing   | Renderer test fails on yellow optimal-line overlay / cyan centerline                                                                                                                      |
| 14  | `p3-s19-impl-renderer-cleanup`            | Implement renderer visual cleanup                                | implementing  | drawOptimalLineGuidance, COLOR_GUIDANCE_LINE_RGB, drawTrackCenterline, COLOR_CENTERLINE removed; only blue/red team guide lines remain                                                    |
| 15  | `p3-s19-green-renderer-cleanup`           | Green validation for renderer visual cleanup                     | green-testing | Red tests now pass; no yellow/cyan assertions remain; no regressions                                                                                                                      |
| 16  | `p3-s19-red-per-car-adaptation`           | Red tests for continuous per-car runtime adaptation              | red-testing   | Test fails when single shared adaptation state mutates every car identically                                                                                                              |
| 17  | `p3-s19-impl-per-car-adaptation`          | Implement per-car runtime adaptation / continuous evolution      | implementing  | Map<carIndex, RuntimeAdaptationState>; no global singleton; browser harness wires each car to its own adaptation entry                                                                    |
| 18  | `p3-s19-green-per-car-adaptation`         | Green validation for continuous per-car evolution                | green-testing | Red tests pass; deterministic probe shows blue/red controllers diverge within 120 frames; no regressions                                                                                  |
| 19  | `p3-s19-red-worker-independent-genomes`   | Red tests for worker evaluation of independent genomes           | red-testing   | Test fails when runner uses one shared network for every car; asserts distinct per-car networks and fitness                                                                               |
| 20  | `p3-s19-impl-worker-independent-genomes`  | Implement per-car genome/network wiring in worker race-pack      | implementing  | createRaceEpisodeRunner accepts one network per car; coevolution container provides one NEAT genome per car; evolution protocol returns per-car/per-team payloads                         |
| 21  | `p3-s19-green-worker-independent-genomes` | Green validation for worker independent-genome evaluation        | green-testing | Red tests pass; no regressions; worker starting-grid baselines updated                                                                                                                    |
| 22  | `p3-s19-green-tier1-divergence-probe`     | Green validation — Tier 1 blue/red divergence probe              | green-testing | Deterministic Tier 1 probe records distinct steering/lateral positions for blue and red within 120 frames; blue follows inner guide, red follows outer guide, no overlap, no cyan divider |

**Final Step 19 validation gate:** 348 tests pass across all focused suites; `npm run lint` clean; `npx tsc --noEmit -p tsconfig.json` clean; `npx tsc --noEmit -p tsconfig.test.json` clean; `npm run build:racing-curriculum` succeeds; `validate-plan-sync` PASS (0 errors, 0 warnings); `validate-plan-phase-packets` PASS (0 errors, 0 warnings).

**Superseded pre-pivot slices (removed from active chain):**

- `p3-s19-red-browser-per-car` — replaced by `p3-s19-red-browser-per-car-controller` plus per-car observation/adaptation slices.
- `p3-s19-impl-browser-per-car` — replaced by `p3-s19-impl-browser-per-car-controller`.
- `p3-s19-green-browser-per-car` — replaced by `p3-s19-green-browser-per-car-controller`.

Their single concern (stop fanning one control to every car) is now enforced by the per-car observation helper, the per-car controller Map, the per-car adaptation Map, and the per-car worker genome wiring.

### Decision Record — DR-011

```yaml
decision_record:
 id: 'DR-011'
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
 result: 'docs/assets/racing-curriculum.bundle.js (731.1kb, 748,645 bytes, )'
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
 last_write_time: ''
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

## Phase 7 — Tier 6: 3v3 advanced strategy — COMPLETED

**Status:** [DONE] — all steps (Step 01 through Step 07) complete. Phase 7 compressed. Analytics-only fallback per DR-008. modeIsEvolvable BLOCKED (nge-core-algorithm ownership). Polyandric reproduction DEFERRED (P1-P5 blockers).

### Step summary (Step 01 – Step 07)

- [DONE] Step 01 — Plan Tier 6 boundary: recorded Tier 6 3v3 advanced-strategy boundary decisions — analytics-only fallback (DR-008: modeIsEvolvable is a declared-but-dead boolean field, no operator reads it, ModulatorBroadcaster/EpisodicSlot/GatingRouter are descriptor-only, no phenotype→Network bridge, entire nge-dna+nge-evolution surface internal-only). Hall-of-fame opponent snapshots: WIRE existing OpponentSnapshotPool primitive. Strategy-divergence analytics: NEW benchmark-local module. Multi-generation evaluation loop: FIX 5 FSM bugs (DR-009). Tire-degradation physics: VERIFY in Step 02. Polyandric reproduction: DEFERRED (P1-P5 carry-forward). Cross-team promotion: NOT APPLICABLE (Tier 6 is terminal). NGE primitive assessment table recorded (14 primitives). DR-008 (analytics-only fallback) and DR-009 (FSM 5-bug fix split into 2 slices) recorded. Step 02-07 packets authored. plan-sync + step-packet gates PASS.
- [DONE] Step 02 — Research hall-of-fame wiring, analytics seams, and NGE dependencies: source-grounded research brief with 9 findings (R1-R9):
- R1 (OpponentSnapshotPool API): `createOpponentSnapshotPool(capacity)`, `addOpponentSnapshot(pool, agentId, payload, frozenAt)` — deep-clone via safeStructuredClone, FIFO eviction, immutable returns. No built-in sample method. Completely unwired into racing worker (zero call sites in examples/).
- R2 (Local opponent-snapshot service): `OpponentSnapshotStore` stores SINGLE snapshot (not pool), dead code. Two incompatible OpponentSnapshot type shapes need adapter: core `{agentId, snapshot, frozenAt}` vs race-pack `{snapshotId, generation, networkPayloads}`.
- R3 (FSM bug — 5 compounding bugs, not 2): (1) handleRaceStep line 334 returns nextState: currentState when runner.frame.done is true, (2) buildGenerationReadyResponse line 396 hardcodes generation:0, (3) advanceTeamGeneration dead code, (4) tryUpdateSnapshot dead code, (5) request-generation line 214 recreates container from scratch destroying all evolution state. Bug 5 is deepest architectural blocker. Additional gaps: carFitnessScores hardcoded to zeros, no genome mutation/selection wired, PHASE_ALLOWED_MESSAGES already allows request-generation in generation-ready.
- R4 (Tire-degradation physics): FULLY IMPLEMENTED — false positive from scout. decayTireState at environment.step.service.ts:108-130 (base decay = |lateral|*0.00012 + |longitudinal|*0.00006 + |speed|*0.000006, worn tires decay faster, clamped [0,1]). gripMultiplier = sqrt(resolveMeanTireHealth) at line 703, applied to steer and reverse throttle. Tested at environment.tier4.test.ts:55-140 (5 tests). Slice 04-s4-tire-physics REMOVED.
- R5 (Strategy-divergence analytics): NEW module (not extension of role-divergence). Interface sketched: StrategyDivergenceSnapshot (per generation: teamAFitness, teamBFitness, pitLapDistributions, reproductionModeMix), StrategyDivergenceClassifierConfig (minGenerations, advantageThreshold, alternationWindow), StrategyDivergenceClassifierResult (isAlternating, dominantPeriod, advantageAmplitude, divergenceScore). Primary export: createStrategyDivergenceTracker(config) with recordSnapshot, classify, getTrajectory. Wiring point: recordSnapshot inside buildGenerationReadyResponse after computing team fitness.
- R6 (3 prerequisite observables): (1) real per-car fitness (replace hardcoded zeros), (2) pit-lap distribution (no pitLap field exists anywhere), (3) reproduction-mode mix (blocked on polyandric). NOT blockers for module scaffold — only for meaningful classifier output.
- R7 (Generation counter): EvolutionProtocolState has NO generation field. TeamPopulationContainer has generation:number (per-team). Add generation:number to EvolutionProtocolState, increment at racing→generation-ready transition.
- R8 (27 carry-forward tsc errors): confirmed in 3 files (coevolution.test.ts, evolution.protocol.test.ts, independent-genomes.test.ts). All duplicate identifier errors from local type redeclarations. Tier 6 tests MUST import from source.
- R9 (Step 04 slice updates): s1 EXPANDED to 5 bugs, s2 ADD type adapter + fitness tracking, s3 ADD pit-lap observable, s4 REMOVED. Delegated to nge-benchmark-scout + boundary-mapper. Cortex index rebuilt from stale.
- [DONE] Step 03 — Red tests for multi-generation loop and analytics contracts: 15 red tests across 3 files:
- `simulation-worker.multi-generation.test.ts` (7 tests): FSM transition (bug 1), runner clearing (bug 2), generation counter (bug 3), advanceTeamGeneration dead code (bugs 4-5), fitness scores, teamABestFitness. All 7/7 fail.
- `simulation-worker.race-pack.tier6.test.ts` (3 tests): OpponentSnapshotPool not in protocol state, snapshots not accumulated, convertCoreToRacePackSnapshot not exported. All 3/3 fail.
- `simulation-worker.strategy-divergence.test.ts` (5 tests): module not found, tracker undefined, classify() undefined, divergenceScore NaN, pit-lap distribution undefined. All 5/5 fail.
- Types imported from source modules (simulation-worker.evolution.types, simulation-worker.coevolution.service, src/neat/nge-collective/neat.nge-collective). Local types only for strategy-divergence (NEW module). Validation commands use --testPathPatterns (plural).
- [DONE] Step 04 — Implement Tier 6 evaluation loop and analytics: 3 implementation slices (04-s4-tire-physics REMOVED):
- `04-s1-fsm-bugfix` [DONE]: Fixed 5 FSM bugs in simulation-worker.evolution.protocol.service.ts + simulation-worker.evolution.types.ts. handleRaceStep transitions to generation-ready when runner.frame.done. buildGenerationReadyResponse uses real generation counter from EvolutionProtocolState (new generation:number field). advanceTeamGeneration called at racing→generation-ready for both teams. request-generation reuses existing coevolution container. carFitnessScores populated from real race finish positions (extractCarFitnessScores). Old createGenerationReadyResponse with hardcoded zeros REMOVED (no dual-path). Coverage closure: 2 tests added for computeFitness path and empty-team guard. 9/9 multi-generation tests pass, 6/6 protocol, 16/16 independent-genomes, 19/19 coevolution. tsc clean, lint 0, build 719.9kb OK.
- `04-s2-hof-wiring` [DONE]: Wired OpponentSnapshotPool into racing coevolution loop. Type adapter convertCoreToRacePackSnapshot bridges core {agentId, snapshot, frozenAt} to race-pack {snapshotId, generation, networkPayloads} in race-pack.service.ts, re-exported from opponent-snapshot.service.ts. tryUpdateSnapshot wired into generation boundary (no longer dead code). advanceTeamGeneration activated. Hall-of-fame snapshots sampled across generations with configurable window. External fitness metadata tracked. OpponentSnapshotPool uses reuse pattern (existingPool ?? createOpponentSnapshotPool). Changed files: evolution.types.ts, evolution.protocol.service.ts, race-pack.service.ts, opponent-snapshot.service.ts, multi-generation.test.ts. 106/106 focused tests pass (tier6 3/3, multi-generation 10/10, coevolution 19/19, opponent-snapshot 9/9, evolution.protocol 6/6, independent-genomes 16/16, race-pack 43/43). 5 expected s3 red failures. tsc clean, lint 0, build OK. plan-sync PASS, cortex-index PASS.
- `04-s3-analytics` [DONE]: Created simulation-worker.strategy-divergence.service.ts with createStrategyDivergenceTracker(config) — recordSnapshot, classify (alternating-advantage classifier: isAlternating, dominantPeriod, advantageAmplitude, divergenceScore), getTrajectory. Pit-lap distribution observable plumbed from race-pack/evaluation layer (new per-car pit-lap counter). StrategyDivergenceSnapshot per generation: teamAFitness, teamBFitness, teamAPitLapDistribution, teamBPitLapDistribution, reproductionModeMix. Reproduction-mode mix sub-metric deferred with polyandric blocker (placeholder). Wiring: recordSnapshot called inside buildGenerationReadyResponse after computing team fitness. Module is separate from role-divergence (zero cross-references). Changed files: strategy-divergence.service.ts, evolution.types.ts, race-pack.service.ts, evolution.protocol.service.ts. 5/5 strategy-divergence tests pass, 10/10 multi-generation, 3/3 tier6, 23/23 coevolution. tsc clean, lint 0, build 719.9kb OK.
- `04-s4-tire-physics` [REMOVED]: Tire-degradation physics already fully implemented (Step 02 R4 confirmed). No action needed.
- [DONE] Step 05 — Green validation and regression triage: 68 suites / 502 tests ALL PASS (0 failures), 3 skipped (polyandric P1/P2). Breakdown: simulation-worker 19 suites/170 passed, browser-entry 32 suites/223 passed, controller 12 suites/70 passed, environment 5 suites/39 passed. tsc (tsconfig.json) clean (exit 0). tsc.test.json 27 carry-forward errors (unchanged — not increased, same 3 files). Lint 0 issues. Build:racing-curriculum OK (719.9kb). plan-sync gate PASS. No regressions from Step 04 changes (FSM bugfix, hof-wiring, analytics, pit-lap plumbing). 3 polyandric tests remain skipped (P1/P2 blockers — pre-existing, not regressions). No src/ files modified by Step 04 — all changes under examples/. coverage-guard N/A.
- [DONE] Step 06 — Document Tier 6 contract: Tier 6 contract documented across 4 source files:
- `simulation-worker.strategy-divergence.service.ts` — Mermaid analytics-flow diagram (flowchart LR) + competitive coevolution Wikipedia citation in module header.
- `simulation-worker.evolution.protocol.service.ts` — Multi-generation evaluation loop Mermaid diagram (flowchart TD), Hall-of-fame opponent snapshot pool section with Coevolution citation, Strategy-divergence analytics section, new extension points table row.
- `simulation-worker.race-pack.service.ts` — 67-line module-level JSDoc header with Mermaid tick-lifecycle diagram (flowchart TD), key concepts, pit-lap distribution observables section, Coevolution citation.
- `simulation-worker.evolution.types.ts` — modeIsEvolvable blocker JSDoc updated with nge-core-algorithm escalation reference.
- Generated: `examples/racing_curriculum/workers/simulation-worker/README.md` regenerated 1258→1739 lines via npm run docs:folders:racing-curriculum. All new sections verified.
- Reference: `examples/racing_curriculum/reference.plans.md` readiness checklist 6 Tier 6 items marked [x].
- Delegation: implementation-executor (×3) for JSDoc, academic-docs-auditor (×1) for citation/Mermaid audit.
- tsc clean, eslint 0 issues, docs exit 0. All documentation atemporal.
- [DONE] Step 07 — Logging and tracker handoff: Phase 7 compressed into this log. Phase 7 marked [DONE]. Carry-forward blockers documented for nge-core-algorithm handoff. phase-compression, log-completion-marker, stale-wip-plans gates run.

**Changed file groups (Phase 7):**

- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.evolution.protocol.service.ts` (FSM 5-bug fix, generation counter, fitness feedback, strategy-divergence wiring)
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.evolution.types.ts` (generation field, modeIsEvolvable JSDoc)
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.opponent-snapshot.service.ts` (OpponentSnapshotPool wiring, type adapter re-export)
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts` (convertCoreToRacePackSnapshot adapter, pit-lap distribution observable)
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.strategy-divergence.service.ts` (NEW — analytics module)
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.multi-generation.test.ts` (7+3 red-green tests)
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.tier6.test.ts` (NEW — 3 HoF/adapter tests)
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.strategy-divergence.test.ts` (NEW — 5 analytics tests)
- `examples/racing_curriculum/workers/simulation-worker/README.md` (regenerated 1258→1739 lines)
- `examples/racing_curriculum/reference.plans.md` (readiness checklist 6 items marked [x])

**Residual risks (carry-forward):**

- P1 (CRITICAL): NGE_DNA adoption gap — racing uses Network, polyandric needs NgeDnaCanonicalEnvelope. Owner: nge-core-algorithm.
- P2 (CRITICAL): NgePolyandricInput/NgePolyandricDroneInput not exported from reproduction.ts. Owner: nge-core-algorithm.
- P3: Racing FSM reproduction step not wired (polyandric call site). Owner: nge-benchmark-workflow.
- P4: Schema mismatch — reference spec uses non-overlapping/queen-weighted, implemented uses roundRobin/byFitness/bySpecialization. Owner: nge-core-algorithm.
- P5: queenBias not honored by merge logic. Owner: nge-core-algorithm.
- DR-008: modeIsEvolvable is a dead boolean field, no operator reads it. ModulatorBroadcaster/EpisodicSlot/GatingRouter are descriptor-only. No phenotype→Network bridge. Owner: nge-core-algorithm.
- 27 tsc.test.json duplicate-identifier errors in 3 test files (coevolution.test.ts, evolution.protocol.test.ts, independent-genomes.test.ts) — pre-existing carry-forward debt.
- 3 polyandric tests remain skipped in simulation-worker.race-pack.tier5.test.ts (lines 179, 188, 197) until P1/P2 resolved.
- cortex-index gate reports stale index (owner: 00-helping).
- Strategy-divergence reproduction-mode mix sub-metric is a placeholder (blocked on polyandric primitive).
- Real per-car fitness, pit-lap distribution, and reproduction-mode mix observables are wired but produce placeholder output until polyandric reproduction is engaged.
