# Public Library Demo-Agnostic Refactor

**Status:** [DONE]

## Active phase/step

Claim: 07-logging @ 2026-07-04T11:47:25-04:00

- Phase 1 — Planning [DONE]
- Step 01: Author step packets and register plan [DONE]
- Phase 2 — Research / Dependency Verification [DONE]
- Step 02: Verify WebGPU dependency state and import boundary [DONE]
- Phase 3 — Implementation [DONE]
- Step 03: Rename critical GPU file, symbols, and demo-specific JSDoc [DONE]
- Phase 4 — Green Validation [DONE]
- Step 04: Run focused green validation [DONE]
- Phase 5 — Documentation [DONE]
- Step 05: Regenerate docs and write migration note [DONE]
- Phase 6 — Logging [DONE]
- Step 06: Compress phase history and close [DONE]

## Scope

Make the NeatapticTS public library (`src/`) demo-agnostic by removing demo-specific file names, exported symbols, JSDoc, internal identifiers, tests, and generated docs. The library should express concepts and features from the library point of view, not from the perspective of any specific demo (`flappy_bird`, `racing_curriculum`, `neatChat`, `asciiMaze`, NGE benchmarks, etc.).

This work is scoped to `src/` only. `examples/` and `docs/browser-tests/` are intentionally demo-specific and must not be refactored.

Full evidence table: `docs/research/public-library-demo-agnostic-audit.md`.

## Migration / Breaking changes

This is a deliberate **breaking public API refactor**. Any external consumer importing from `src/architecture/network/gpu/network.gpu.racing.ts` or using the `Racing*` / `evaluateRacing*` symbols will break.

Proposed renames (final names to be confirmed in the migration note):

| Old                                                       | New                                                                 |
| --------------------------------------------------------- | ------------------------------------------------------------------- |
| `src/architecture/network/gpu/network.gpu.racing.ts`      | `src/architecture/network/gpu/network.gpu.batch-evaluation.ts`      |
| `RacingBatchOptions`                                      | `BatchEvaluationOptions`                                            |
| `evaluateRacingGeneration`                                | `evaluateBatchGeneration`                                           |
| `RacingAgentRequest`                                      | `AgentEvaluationRequest`                                            |
| `evaluateConcurrentRacingAgents`                          | `evaluateConcurrentAgents`                                          |
| `src/architecture/network/gpu/network.gpu.racing.test.ts` | `src/architecture/network/gpu/network.gpu.batch-evaluation.test.ts` |

The final Phase 5 slice will add a migration note to `RELEASE.md` mapping every removed public symbol to its replacement. No backward-compatibility shims or re-export wrappers will be kept; old names must be fully removed in the same slice that introduces the replacements (no deferred cleanup).

## Dependencies / Cross-plan coordination

- **Active blocker:** `plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md` is currently [WIP] and actively editing `src/architecture/network/gpu/network.gpu.racing.ts` / `network.gpu.racing.test.ts` (stub exports `RacingAgentRequest` and `evaluateConcurrentRacingAgents`). The demo-agnostic refactor must not start Phase 2 implementation until the WebGPU plan has completed its current Phase 2 slice and the racing-named GPU surface is stable.
- **Coordination rule:** if the WebGPU plan lands new racing-named symbols in `network.gpu.racing.ts`, this plan must absorb them into the same rename pass.
- **Downstream consumers:** `examples/` does not import the racing-named GPU module directly, so no internal demo migration is required. External consumers are covered by the breaking-change migration note.
- **Generated docs:** `src/**/README.md` files must be regenerated via `npm run docs` after source JSDoc edits. Do not hand-edit generated READMEs.

## Coverage backlog

- [x] Research audit recorded in `docs/research/public-library-demo-agnostic-audit.md`.
- [x] Boundary map produced by `boundary-mapper` for the critical GPU rename.
- [x] Acceptance criteria drafted by `acceptance-criteria-writer`.
- [x] Plan registered in `plans/README.md` and `plans/Roadmap.md` with machine-readable step packets and passing gate checks.
- [x] Phase 2 dependency verification.
- [x] Phase 3 critical rename and JSDoc sanitization (slice 03-01 [DONE]; 03-02 [DONE]).
- [x] Phase 3 slice 03-03 source-file and co-located test-file JSDoc sanitization [DONE].
- [x] Phase 3 slice 03-04 NGE source-file JSDoc sanitization [DONE].
- [x] Phase 3 slice 03-05 two-population internal identifier rename [DONE].
- [x] Phase 3 slice 03-06 export/viz/worker JSDoc sanitization [DONE].
- [x] Phase 3 slice 03-07 test fixtures and comments sanitization [DONE].
- [x] Phase 3 implementation COMPLETE — all 7 slices [DONE].
- [x] Phase 4 green validation [DONE].
- [x] Phase 5 docs regeneration and migration note [DONE].
- [x] Phase 6 logging / closure [DONE].

## Final state

All 6 phases are [DONE]. The plan/log pair has been archived to `plans/completed/`. `plans/README.md` and `plans/Roadmap.md` reflect the completed state. Closure gates (`phase-compression`, `log-completion-marker`, `stale-wip-plans`) pass. The unrelated WebGPU real-performance plan remains active and untouched.

## Latest validation evidence

- `npx tsc --noEmit -p tsconfig.json` — exit 0, 0 diagnostics.
- `npm run lint` — exit 0, 0 errors.
- Focused Jest slices (using `--testPathPatterns`, plural) — all pass:
  - `src/architecture/network/gpu/network.gpu.batch-evaluation` — pass.
  - `src/neat/nge-collective/neat.nge-collective.two-population` — pass.
  - `src/neat/export/neat.export.test.ts` — pass.
  - `src/visualization/network-view/network-view.test.ts` — pass.
  - `src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts` — pass.
- `npm run quality:folder -- --folder=src/architecture/network/gpu` — PASS, 0 smells.
- `npm run quality:folder -- --folder=src/neat/nge-collective` — PASS, 0 smells.
- Tier-1 gates — all `pass: true`:
  - `step-packet.gate.mjs`
  - `plan-sync.gate.mjs`
  - `agent-graph.gate.mjs`
  - `learning-event.gate.mjs`
- Broad targeted coverage run — 313 suites / 4042 tests passed; all 15 executable touched `src/` files at 100% statements/branches/functions/lines.
- Two tooling/coverage gaps were closed during validation:
  - Added one owner-local test in `src/neat/nge-evolution/neat.nge-evolution.reproduction.queen-bias.test.ts` to exercise the `seedPolicy === 'queen-weighted'` branch (delegated to `coverage-guard`).
  - Repaired `scripts/folder-quality-metrics.mjs` to load ambient `.d.ts` declarations and git-scope the missing-test heuristic (delegated to `helping-gap-resolution-coordinator`).
  - Repaired `rag-index/build-browser-snapshot.mjs` to emit `generated_at` so the `cortex-index` gate can verify snapshot currency.
- `cortex-index` gate — `pass: true` after rebuilding the semantic index and snapshot.
- Generated README regeneration:
  - `npm run docs` — exit 0; regenerated all `src/**/README.md` and `dist-docs/` outputs.
  - `git grep -Ein "(flappy|flappy_bird|flappy-bird|asciiMaze|ASCII[ _-]?Maze|neatchat|racing_curriculum|racing-curriculum|ant[ _-]?hive|predator/prey|RacingBatchOptions|evaluateRacingGeneration|network\.gpu\.racing)" src/README.md src/**/README.md` — no matches; generated docs are demo-free.
  - `src/architecture/network/gpu/README.md` references `network.gpu.batch-evaluation.ts`, `BatchEvaluationOptions`, `evaluateBatchGeneration`, `AgentEvaluationRequest`, and `evaluateConcurrentAgents`.
  - `src/architecture/network/gpu/docs.order.json` references `network.gpu.batch-evaluation.ts`.
- Migration note:
  - Added to top of `RELEASE.md`, mapping `network.gpu.racing.ts`, `RacingBatchOptions`, `evaluateRacingGeneration`, `RacingAgentRequest`, `evaluateConcurrentRacingAgents` to their batch-evaluation replacements.
  - Includes internal two-population renames (`raceState` → `episodeState`, `car` → `agent`, `race-pack` → `episode-pack`) and JSDoc sanitization scope.
- Post-doc quality checks:
  - `npx tsc --noEmit -p tsconfig.json` — pass.
  - `npm run lint` — pass.
  - `npx prettier --write RELEASE.md` — pass.
- Tier-1 gates — all `pass: true`:
  - `step-packet.gate.mjs`
  - `plan-sync.gate.mjs`
  - `plan-slice-quality.gate.mjs`
  - `agent-graph.gate.mjs`
  - `learning-event.gate.mjs`

---

### Phase 1 — Planning [DONE]

**Phase objective:** Author the step packets for this refactor, register the plan in the index and roadmap, and confirm acceptance criteria.

[DONE] Step 01: authored step packets, registered plan, acceptance criteria confirmed, planning gates passed. Full packet and validation evidence are in `plans/Public_Library_Demo_Agnostic_Refactor.logs.md`.

---

### Phase 2 — Research / Dependency Verification [DONE]

**Phase objective:** Confirm the WebGPU plan is no longer actively editing the racing-named GPU surface, and re-verify the import boundary.

[DONE] Step 02: WebGPU dependency state verified, import boundary confirmed, boundary map recorded. Full packet and validation evidence are in `plans/Public_Library_Demo_Agnostic_Refactor.logs.md`.

---

### Phase 3 — Implementation [DONE]

**Phase objective:** Execute the demo-agnostic refactor in `src/` using green-only TDD.

[DONE] Step 03: all 7 slices green — renamed GPU batch-evaluation module and symbols, sanitized GPU/eval-pack/NGE/export/viz/worker JSDoc, renamed two-population identifiers, updated test fixtures. Full step packet, PlanUpdate records, and validation evidence are in `plans/Public_Library_Demo_Agnostic_Refactor.logs.md`.

---

### Phase 4 — Green Validation [DONE]

**Phase objective:** Confirm the refactor compiles, lints, and passes targeted tests for every affected folder.

[DONE] Step 04: focused green validation passed — tsc, lint, targeted Jest slices, folder-quality gates, and 100% coverage on all touched `src/` files. Full step packet and validation evidence are in `plans/Public_Library_Demo_Agnostic_Refactor.logs.md`.

---

### Phase 5 — Documentation [DONE]

**Phase objective:** Regenerate generated docs from sanitized source and write the breaking-change migration note.

[DONE] Step 05: docs regenerated via `npm run docs`, generated READMEs are demo-free, migration note added to `RELEASE.md`, final typecheck/lint/prettier passed. Full step packet and validation evidence are in `plans/Public_Library_Demo_Agnostic_Refactor.logs.md`.

---

### Phase 6 — Logging [DONE]

```yaml
phase: 6
title: Logging
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/completed/Public_Library_Demo_Agnostic_Refactor.plans.md
copy_paste: true
next_phase: null
skills:
  - plan-alignment
validation:
  - 'node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/completed/Public_Library_Demo_Agnostic_Refactor.plans.md'
acceptance_criteria:
  - 'Plan phase/step YAML blocks pass the step-packet gate.'
placeholder_steps:
  - 'Step 06 — Compress phase history and close'
```

**Phase objective:** Compress completed phase history and close the workstream once all green validation is recorded.

#### Step 06: Compress phase history and close [DONE]

```yaml
phase: 6
step: 6
title: 'Compress phase history and close'
status: '[DONE]'
goal: logging
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/Public_Library_Demo_Agnostic_Refactor.plans.md
copy_paste: true
next_step: null
skills:
  - tracker-handoff
  - capturing-learning-event
validation:
  - 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
  - 'node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json'
  - 'node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json'
acceptance_criteria:
  - 'All phases are marked [DONE] in the plan tracker'
  - 'Phase compression and stale-wip-plans gates pass before archive'
  - 'Closed plan/log pair is moved to plans/completed/'
```

**User instruction:** Paste this full step packet.

**Step objective:** Close the workstream cleanly: compress history, create/update the matching `.logs.md`, move the plan/log pair to `plans/completed/`, and ensure no stale WIP marker remains.

**Context the agent must know:**

- Closure requires the `phase-compression`, `log-completion-marker`, and `stale-wip-plans` gates.
- Learning events are append-only and must not contain timestamps or session identifiers.

**Execution steps:**

1. Mark all prior phases `[DONE]`.
2. Compress each phase's detailed transcript into a single coverage note.
3. Create `plans/Public_Library_Demo_Agnostic_Refactor.logs.md` with durable done-state records.
4. Move the `.plans.md` and `.logs.md` pair to `plans/completed/`.
5. Update `plans/README.md` and `plans/Roadmap.md` to remove active entries or mark [DONE].
6. Run closure gates.

**Stop conditions:**

- Done when all closure gates pass.
- Blocked if any gate fails; fix and rerun.

**Required validation:**

- `node scripts/agent-customization/gates/phase-compression.gate.mjs --json`
- `node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json`
- `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json`

**Plan update requirement:** Record final closure evidence and remove the `Handoff query` section unless reopen guidance is explicitly requested.

---
