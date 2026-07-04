# Public Library Demo-Agnostic Refactor Log

**Status:** [DONE] — workstream closed; all 6 phases complete.

This log holds the compressed detailed history for completed Phases 1–6 of `plans/Public_Library_Demo_Agnostic_Refactor.plans.md`. The plan file has been archived to `plans/completed/Public_Library_Demo_Agnostic_Refactor.plans.md`.

---

## Phase 1 — Planning [DONE]

[DONE] Phase 1: authored step packets, registered plan in README/Roadmap, acceptance criteria drafted, planning gates passed. Full step packet and validation evidence preserved below.

### Phase 1 — Planning [DONE]

**Phase objective:** Author the step packets for this refactor, register the plan in the index and roadmap, and confirm acceptance criteria.

**Phase progression rule:** Start with only Step 01. Step 01 authors the remaining numbered step packets before the phase can advance.

#### Step 01: Author step packets and register plan [DONE]

```yaml
phase: 1
step: 1
title: 'Author step packets and register plan'
goal: 'planning'
status: '[DONE]'
expansion: none
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Public_Library_Demo_Agnostic_Refactor.plans.md'
copy_paste: true
next_step: 'Step 02 — Verify WebGPU dependency state and import boundary'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
  - 'phase-handoff-workflow'
  - 'planning-acceptance-criteria'
validation:
  - 'node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md'
  - 'node scripts/agent-customization/gates/plan-sync.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md'
  - 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md'
acceptance_criteria:
  - id: AC-PLAN-001
    text: 'Plan file exists in plans/ with machine-readable phase and step YAML blocks'
    validation: 'node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md'
  - id: AC-PLAN-002
    text: 'Plan is registered in plans/README.md and plans/Roadmap.md with consistent status'
    validation: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md'
  - id: AC-PLAN-003
    text: 'All implementation slices are ≤4 hours and include acceptance criteria'
    validation: 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md'
constitution_check:
  - 'principle-4-breadth-first-recoverable'
  - 'principle-5-unique-ids'
```

**Step objective:** Produce a complete, executable plan for removing demo-specific naming from `src/`, including step packets, slices, acceptance criteria, migration notes, and cross-plan dependency coordination.

**Context the agent must know:**

- Full evidence table is in `docs/research/public-library-demo-agnostic-audit.md`.
- The WebGPU plan is actively editing `src/architecture/network/gpu/network.gpu.racing.ts`; this plan must sequence after it.
- `examples/` and `docs/browser-tests/` are out of scope.
- Generated READMEs are produced by `npm run docs`.

**Execution steps:**

1. Read the research audit and identify severity clusters.
2. Delegate boundary mapping and acceptance-criteria drafting to specialists.
3. Author Step 02–06 packets with slices.
4. Add migration/breaking-changes section.
5. Update `plans/README.md` and `plans/Roadmap.md`.
6. Run `plan-sync`, `step-packet`, and `plan-slice-quality` gates.

**Stop conditions:**

- Done when the plan file passes all three gates.
- Blocked if WebGPU dependency cannot be verified.

**Required validation:**

- `node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md`
- `node scripts/agent-customization/gates/plan-sync.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md`
- `node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md`

**Plan update requirement:** Record gate outputs under `## Latest validation evidence` and set Step 02 to `[WIP]` with the WebGPU dependency noted.

---

---

## Phase 2 — Research / Dependency Verification [DONE]

[DONE] Phase 2: WebGPU dependency state verified, import boundary confirmed via git grep, boundary map recorded. Full step packet and validation evidence preserved below.

### Phase 2 — Research / Dependency Verification [DONE]

```yaml
phase: 2
title: 'Research / Dependency Verification'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/Public_Library_Demo_Agnostic_Refactor.plans.md
copy_paste: true
next_phase: Implementation
skills:
  - plan-alignment
research_artifact: docs/research/public-library-demo-agnostic-import-boundary-map.md
validation:
  - 'node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md'
  - 'git -C C:\\NeatapticTS status --short -- src/architecture/network/gpu/network.gpu.racing.ts src/architecture/network/gpu/network.gpu.racing.test.ts'
acceptance_criteria:
  - 'Step 02 validation confirms the racing-named GPU surface is stable and the plan schema still passes the step-packet gate.'
placeholder_steps:
  - 'Step 02 — Verify WebGPU dependency state and import boundary'
```

**Phase objective:** Confirm the WebGPU plan is no longer actively editing the racing-named GPU surface, and re-verify the import boundary so the rename does not conflict with in-flight work.

#### Step 02: Verify WebGPU dependency state and import boundary [DONE]

```yaml
phase: 2
step: 2
title: 'Verify WebGPU dependency state and import boundary'
status: '[DONE]'
goal: researching
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/Public_Library_Demo_Agnostic_Refactor.plans.md
copy_paste: true
next_step: 'Step 03 — Rename critical GPU file and exported symbols'
skills:
  - research-methodology
  - plan-alignment
research_artifact: docs/research/public-library-demo-agnostic-import-boundary-map.md
validation:
  - 'git -C C:\\NeatapticTS status --short -- src/architecture/network/gpu/network.gpu.racing.ts src/architecture/network/gpu/network.gpu.racing.test.ts'
  - 'git -C C:\\NeatapticTS grep -n "network.gpu.racing\\|RacingBatchOptions\\|evaluateRacingGeneration\\|RacingAgentRequest\\|evaluateConcurrentRacingAgents" -- src/ examples/'
acceptance_criteria:
  - 'WebGPU plan is not actively modifying network.gpu.racing.ts or its test file'
  - 'No new src/ or examples/ consumers of racing-named GPU symbols exist'
specialists:
  - plan-scout
```

**Step objective:** Verify that the refactor can safely rename `src/architecture/network/gpu/network.gpu.racing.ts` without colliding with the active WebGPU plan, and confirm the import graph remains an isolated public API leaf.

**Context the agent must know:**

- Downstream plan: `plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md`.
- Prior boundary map shows no `examples/` consumers and only owner-local test imports.
- If WebGPU is still mid-slice, this step must return a blocker and wait.

**Execution steps:**

1. Read the WebGPU plan's active phase/step to confirm status.
2. Check git status for uncommitted edits to `network.gpu.racing.ts` / `network.gpu.racing.test.ts`.
3. Re-run the git grep boundary query.
4. Update this plan with the dependency verdict.

**Stop conditions:**

- Done when WebGPU dependency is clear and boundary is unchanged.
- Blocked if WebGPU is still editing the racing file; escalate to `00.cross-tier-helper` with a decision record.

**Required validation:**

- `git status --short -- src/architecture/network/gpu/network.gpu.racing.ts src/architecture/network/gpu/network.gpu.racing.test.ts`
- `git grep -n "network.gpu.racing\|RacingBatchOptions\|evaluateRacingGeneration\|RacingAgentRequest\|evaluateConcurrentRacingAgents" -- src/ examples/`

**Dependency verdict:**

- WebGPU slice `02-05b-buffer-parallel` is paused by user directive; the racing file has uncommitted edits from that completed (functionally green) slice, but no new active work is landing.
- Cortex-first search + `git grep` confirm **zero** `src/` or `examples/` files import from `network.gpu.racing.ts` outside the co-located test.
- Only references outside the module are a JSDoc mention in `src/architecture/network/gpu/network.gpu.batched.ts`, generated GPU READMEs, and `src/architecture/network/gpu/docs.order.json`.
- Boundary map recorded in `docs/research/public-library-demo-agnostic-import-boundary-map.md`.

**Plan update requirement:** Record the dependency verdict, advance Step 03 to `[WIP]` if clear, or keep Step 02 `[WIP]` with a blocker note.

---

---

## Phase 3 — Implementation [DONE]

[DONE] Phase 3: all 7 slices green — file rename (03-01), symbol rename (03-02), GPU/eval-pack JSDoc (03-03), NGE JSDoc (03-04), internal identifiers (03-05), export/viz/worker JSDoc (03-06), test fixtures (03-07). Detailed PlanUpdate records and validation evidence preserved below.

### PlanUpdate slice records

## PlanUpdate: slice 03-01

```yaml
PlanUpdate:
  slice_id: 03-01
  changed_files:
    - src/architecture/network/gpu/network.gpu.batch-evaluation.ts
    - src/architecture/network/gpu/network.gpu.batch-evaluation.test.ts
    - src/architecture/network/gpu/docs.order.json
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check src/architecture/network/gpu/network.gpu.batch-evaluation.ts src/architecture/network/gpu/network.gpu.batch-evaluation.test.ts src/architecture/network/gpu/docs.order.json'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/architecture/network/gpu/network.gpu.batch-evaluation'
  rollback:
    - 'git mv src/architecture/network/gpu/network.gpu.batch-evaluation.ts src/architecture/network/gpu/network.gpu.racing.ts'
    - 'git mv src/architecture/network/gpu/network.gpu.batch-evaluation.test.ts src/architecture/network/gpu/network.gpu.racing.test.ts'
    - 'git checkout -- src/architecture/network/gpu/docs.order.json'
  next: 'Dispatch 04-implementing for slice 03-02 (rename exported Racing* symbols) then 05-green-testing'
  validation_evidence:
    - 'git ls-files src/architecture/network/gpu/network.gpu.racing.ts: (empty)'
    - 'git ls-files src/architecture/network/gpu/network.gpu.batch-evaluation.ts: present'
    - 'git ls-files src/architecture/network/gpu/*.test.ts | grep -i racing: (empty)'
    - 'docs.order.json fileOrder includes network.gpu.batch-evaluation.ts'
    - 'tsc: OK'
    - 'lint: 0 issues'
    - 'prettier: 0 issues'
    - '05-green-testing: npx jest --testPathPatterns=src/architecture/network/gpu/network.gpu.batch-evaluation --no-coverage: 12/12 pass, exit 0'
    - '05-green-testing: npx jest --testPathPatterns=src/architecture/network/gpu/network.gpu.batched --no-coverage: 27/27 pass, exit 0 (no regression)'
    - '05-green-testing: focused coverage on renamed file: 100% Stmts / 100% Branch / 100% Funcs / 100% Lines'
    - 'coverage-guard delegated: confirmed 100% coverage for src/architecture/network/gpu/network.gpu.batch-evaluation.ts'
    - 'code-quality-auditor delegated: tsc production+test, lint, prettier all pass'
    - 'plan-sync gate: pass'
    - 'agent-graph gate: pass'
    - 'step-packet gate: pass'
    - 'GPU Real-Device Gate: waived per task packet (pure file rename, exported symbols unchanged for slice 03-01)'
```

## PlanUpdate: slice 03-02

```yaml
PlanUpdate:
  slice_id: 03-02
  changed_files:
    - src/architecture/network/gpu/network.gpu.batch-evaluation.ts
    - src/architecture/network/gpu/network.gpu.batch-evaluation.test.ts
    - src/architecture/network/gpu/network.gpu.batched.ts
    - src/architecture/network/gpu/README.md
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check src/architecture/network/gpu/network.gpu.batch-evaluation.ts src/architecture/network/gpu/network.gpu.batch-evaluation.test.ts src/architecture/network/gpu/network.gpu.batched.ts'
    - 'git grep -Ein "RacingBatchOptions|evaluateRacingGeneration|RacingAgentRequest|evaluateConcurrentRacingAgents" src/'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/architecture/network/gpu/network.gpu.batch-evaluation'
  rollback:
    - 'git mv src/architecture/network/gpu/network.gpu.batch-evaluation.ts src/architecture/network/gpu/network.gpu.racing.ts'
    - 'git mv src/architecture/network/gpu/network.gpu.batch-evaluation.test.ts src/architecture/network/gpu/network.gpu.racing.test.ts'
    - 'git checkout -- src/architecture/network/gpu/network.gpu.batched.ts src/architecture/network/gpu/README.md'
  next: 'Dispatch 05-green-testing for the focused test slice, then proceed to slice 03-03'
  validation_evidence:
    - 'AC-03-02-001: git grep -Ein "RacingBatchOptions|evaluateRacingGeneration|RacingAgentRequest|evaluateConcurrentRacingAgents" src/ → ZERO_HITS (exit 1, no matches)'
    - 'AC-03-02-002: git grep -Ein "evaluateRacingGeneration" src/architecture/network/gpu/network.gpu.batched.ts → ZERO_HITS (exit 1, no matches)'
    - 'AC-03-02-003: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=network.gpu.batch-evaluation --no-coverage → 12/12 pass, exit 0'
    - 'Regression: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=network.gpu.batched --no-coverage → 27/27 pass, exit 0'
    - 'Focused coverage: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=network.gpu.batch-evaluation --coverage --collectCoverageFrom="src/architecture/network/gpu/network.gpu.batch-evaluation.ts" → 100% Stmts / 100% Branch / 100% Funcs / 100% Lines'
    - 'coverage-guard delegated: src/architecture/network/gpu/network.gpu.batch-evaluation.ts and network.gpu.batched.ts both 100% Stmts / 100% Branch / 100% Funcs / 100% Lines'
    - 'code-quality-auditor delegated: tsc (tsconfig.json + tsconfig.test.json), npm run lint, npx eslint on changed .ts files, npx prettier --check on changed .ts files → all pass'
    - 'npx tsc --noEmit -p tsconfig.json: exit 0'
    - 'npx tsc --noEmit -p tsconfig.test.json: exit 0'
    - 'npm run lint: exit 0, 0 issues'
    - 'npx eslint on changed .ts files: exit 0, 0 issues'
    - 'npx prettier --check on changed .ts files: exit 0, 0 issues'
    - 'plan-sync gate: pass'
    - 'agent-graph gate: pass'
    - 'step-packet gate: pass'
    - 'learning-event gate: pass'
    - 'GPU Real-Device Gate: waived per task packet (symbol rename only, no logic changes, no GPU runtime change)'
    - 'Workflow gap: cortex-index gate fails because build-browser-snapshot.mjs does not write the generated_at field expected by cortex-index.gate.mjs. Recorded as learning event; not a slice 03-02 blocker.'
    - 'Plan tracker updated: slice 03-02 status [PLANNED] -> [DONE]; active_slice 03-02 -> 03-03.'
    - 'Re-run step-packet gate after tracker update: pass'
    - 'Re-run plan-sync gate after tracker update: pass'
```

## PlanUpdate: slice 03-03

```yaml
PlanUpdate:
  slice_id: 03-03
  changed_files:
    - src/architecture/network/gpu/network.gpu.batch-evaluation.ts
    - src/architecture/network/gpu/network.gpu.batched.ts
    - src/architecture/network/gpu/network.gpu.parity.test.ts
    - src/architecture/network/evaluation-pack/network.evaluation-pack.ts
    - src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
    - 'npx prettier --check src/architecture/network/gpu/network.gpu.batch-evaluation.ts src/architecture/network/gpu/network.gpu.batched.ts src/architecture/network/gpu/network.gpu.parity.test.ts src/architecture/network/evaluation-pack/network.evaluation-pack.ts src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts'
    - 'git grep -Ein "racing-curriculum|racing generation|racing eligibility|racing agents|racing-browser|demo" src/architecture/network/gpu/*.ts'
    - 'git grep -Ein "racing, predator/prey, and ant-hive|RacingRenderFrame|track physics|racing-specific" src/architecture/network/evaluation-pack/*.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/architecture/network/gpu/network.gpu.batch-evaluation'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/architecture/network/gpu/network.gpu.parity'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/architecture/network/evaluation-pack/network.evaluation-pack'
  rollback:
    - 'git checkout -- src/architecture/network/gpu/network.gpu.batch-evaluation.ts src/architecture/network/gpu/network.gpu.batched.ts src/architecture/network/gpu/network.gpu.parity.test.ts src/architecture/network/evaluation-pack/network.evaluation-pack.ts src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts'
  next: 'Dispatch 05-green-testing for the focused test slices, then proceed to slice 03-04'
  validation_evidence:
    - 'AC-03-03-001: git grep -Ein "racing-curriculum|racing generation|racing eligibility|racing agents|racing-browser|demo" src/architecture/network/gpu/*.ts → ZERO_HITS (exit 1, no matches)'
    - 'AC-03-03-002: git grep -Ein "racing, predator/prey, and ant-hive|RacingRenderFrame|track physics|racing-specific" src/architecture/network/evaluation-pack/*.ts → ZERO_HITS (exit 1, no matches)'
    - 'Focused test: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.batch-evaluation → 12/12 pass, exit 0'
    - 'Focused test: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.parity → 13/13 pass across 2 suites, exit 0'
    - 'Focused test: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/evaluation-pack/network.evaluation-pack → 8/8 pass, exit 0'
    - 'Focused coverage: npx jest --testPathPatterns=network.gpu.batch-evaluation.test.ts --coverage --collectCoverageFrom="src/architecture/network/gpu/network.gpu.batch-evaluation.ts" → 100% Stmts / 100% Branch / 100% Funcs / 100% Lines'
    - 'Focused coverage: npx jest --testPathPatterns=network.gpu.batched.test.ts --coverage --collectCoverageFrom="src/architecture/network/gpu/network.gpu.batched.ts" → 100% Stmts / 100% Branch / 100% Funcs / 100% Lines'
    - 'Focused coverage: npx jest --testPathPatterns=network.evaluation-pack.test.ts --coverage --collectCoverageFrom="src/architecture/network/evaluation-pack/network.evaluation-pack.ts" → 100% Stmts / 100% Branch / 100% Funcs / 100% Lines'
    - 'code-quality-auditor delegated: npx tsc --noEmit -p tsconfig.json, npx tsc --noEmit -p tsconfig.test.json, npm run lint, npx prettier --check on 5 slice .ts files → all pass, 0 issues'
    - 'coverage-guard delegated: src/architecture/network/gpu/network.gpu.batch-evaluation.ts, network.gpu.batched.ts, and src/architecture/network/evaluation-pack/network.evaluation-pack.ts all at 100% Stmts/Branch/Funcs/Lines'
    - 'plan-sync gate: pass (re-run after plan edit)'
    - 'step-packet gate: pass (re-run after plan edit)'
    - 'plan-slice-quality gate: pass (re-run after plan edit)'
    - 'agent-graph gate: pass'
    - 'learning-event gate: pass'
    - 'GPU Real-Device Gate: waived per task packet (comment-only changes in test files, no GPU logic changes; source-file JSDoc changes are doc-only)'
  blockers: []
```

## PlanUpdate: slice 03-04

```yaml
PlanUpdate:
  slice_id: 03-04
  changed_files:
    - src/neat/neat.nge-lifecycle.ts
    - src/neat/nge-collective/neat.nge-collective.shared-field.ts
    - src/neat/nge-collective/neat.nge-collective.team-fitness.ts
    - src/neat/nge-collective/neat.nge-collective.ts
    - src/neat/nge-collective/neat.nge-collective.types.ts
    - src/neat/nge-dna/neat.nge-dna.ts
    - src/neat/nge-dna/neat.nge-dna.types.ts
    - src/neat/nge-evolution/neat.nge-evolution.reproduction.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check src/neat/neat.nge-lifecycle.ts src/neat/nge-collective/neat.nge-collective.shared-field.ts src/neat/nge-collective/neat.nge-collective.team-fitness.ts src/neat/nge-collective/neat.nge-collective.ts src/neat/nge-collective/neat.nge-collective.types.ts src/neat/nge-dna/neat.nge-dna.ts src/neat/nge-dna/neat.nge-dna.types.ts src/neat/nge-evolution/neat.nge-evolution.reproduction.ts src/neat/nge-juvenile/neat.nge-juvenile.ts'
    - 'git grep -Ein "(racing|ant-hive|predator/prey|racing-worker)" src/neat/nge-*.ts src/neat/nge-*/*.ts src/neat/neat.nge-lifecycle.ts'
    - 'git grep -Ein "Racing|Ant Hive" src/neat/nge-collective/*.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-collective/neat.nge-collective.team-fitness'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-dna/neat.nge-dna'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-evolution/neat.nge-evolution.reproduction'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-collective/neat.nge-collective'
  rollback:
    - 'git checkout -- src/neat/neat.nge-lifecycle.ts src/neat/nge-collective/neat.nge-collective.shared-field.ts src/neat/nge-collective/neat.nge-collective.team-fitness.ts src/neat/nge-collective/neat.nge-collective.ts src/neat/nge-collective/neat.nge-collective.types.ts src/neat/nge-dna/neat.nge-dna.ts src/neat/nge-dna/neat.nge-dna.types.ts src/neat/nge-evolution/neat.nge-evolution.reproduction.ts src/neat/nge-juvenile/neat.nge-juvenile.ts'
  next: 'Dispatch 05-green-testing for the focused test slices, then proceed to slice 03-05'
  validation_evidence:
    - 'AC-03-04-001: git grep -Ein "(racing|ant-hive|predator/prey|racing-worker)" src/neat/nge-*.ts src/neat/nge-*/*.ts src/neat/neat.nge-lifecycle.ts -> ZERO_HITS in slice files; only non-slice file src/neat/nge-collective/neat.nge-collective.two-population.ts still contains "racing" (handled in slice 03-05)'
    - 'AC-03-04-002: git grep -Ein "Racing|Ant Hive" src/neat/nge-collective/*.ts -> ZERO_HITS in slice files; only non-slice file neat.nge-collective.two-population.ts still contains "racing" (handled in slice 03-05)'
    - 'npx tsc --noEmit -p tsconfig.json: PASS'
    - 'npm run lint: PASS (0 issues)'
    - 'npx prettier --check on 9 changed .ts files: PASS (all use Prettier code style)'
    - 'GPU Real-Device Gate: waived (JSDoc-only slice, no GPU logic changes)'
  blockers: []
```

## PlanUpdate: slice 03-04 green validation

```yaml
PlanUpdate:
  slice_id: 03-04
  role: 05-green-testing
  validation_run:
    - command: 'git grep -Ein "(racing|ant-hive|predator/prey|racing-worker)" src/neat/nge-*.ts src/neat/nge-*/*.ts src/neat/neat.nge-lifecycle.ts'
      result: 'ZERO_HITS in slice 03-04 files; only src/neat/nge-collective/neat.nge-collective.two-population.ts (slice 03-05) still contains "racing"'
      gate: AC-03-04-001
      pass: true
    - command: 'git grep -Ein "Racing|Ant Hive" src/neat/nge-collective/*.ts'
      result: 'ZERO_HITS in slice 03-04 files; only neat.nge-collective.two-population.ts (slice 03-05) still contains "Racing"'
      gate: AC-03-04-002
      pass: true
    - command: 'npx tsc --noEmit -p tsconfig.json'
      result: 'PASS (exit 0, 0 errors)'
      pass: true
    - command: 'npm run lint'
      result: 'PASS (0 issues)'
      pass: true
    - command: 'npx prettier --check <9 changed .ts files>'
      result: 'PASS (all matched files use Prettier code style)'
      pass: true
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-collective --coverage --collectCoverageFrom="src/neat/nge-collective/**/*.ts" --runInBand'
      result: 'PASS — 7 suites, 89 tests, 100% statements/branches/functions/lines for nge-collective files'
      pass: true
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-dna --coverage --collectCoverageFrom="src/neat/nge-dna/**/*.ts" --runInBand'
      result: 'PASS — 5 suites, 99 tests, 100% statements/branches/functions/lines for nge-dna files'
      pass: true
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-evolution --coverage --collectCoverageFrom="src/neat/nge-evolution/**/*.ts" --runInBand'
      result: 'PASS — 5 suites, 53 tests; neat.nge-evolution.reproduction.ts at 99.67% statements, 99.23% branches, 100% functions, 99.67% lines (line 796 queen-weighted branch pre-existing gap, unchanged from baseline 99.34/98.61/100/99.33 in coverage/coverage-summary.json, not a regression from JSDoc-only edits)'
      pass: true
      coverage_note: 'pre-existing gap in changed file, not introduced by this slice'
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-juvenile --coverage --collectCoverageFrom="src/neat/nge-juvenile/**/*.ts" --runInBand'
      result: 'PASS — 4 suites, 155 tests, 100% statements/branches/functions/lines for nge-juvenile files'
      pass: true
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/neat.nge-lifecycle --coverage --collectCoverageFrom="src/neat/neat.nge-lifecycle.ts" --runInBand'
      result: 'PASS — 2 suites, 9 tests, 100% statements/branches/functions/lines for neat.nge-lifecycle.ts'
      pass: true
  workflow_gates:
    - gate: plan-sync
      pass: true
      owner: validate-plan-sync.mjs
    - gate: step-packet
      pass: true
      owner: step-packet.gate.mjs
    - gate: agent-graph
      pass: true
      owner: validate-agent-graph.mjs
    - gate: learning-event
      pass: true
      owner: .github/ai-learning/learning-log.jsonl
  gpu_real_device_gate: 'waived — JSDoc-only slice, no GPU logic changes'
  blockers: []
  next: 'Slice 03-04 is green; proceed to slice 03-05 implementation or Phase 4 green validation as orchestrated'
```

## PlanUpdate: slice 03-05

```yaml
PlanUpdate:
  slice_id: 03-05
  changed_files:
    - src/neat/nge-collective/neat.nge-collective.two-population.ts
    - src/neat/nge-collective/neat.nge-collective.two-population.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check src/neat/nge-collective/neat.nge-collective.two-population.ts src/neat/nge-collective/neat.nge-collective.two-population.test.ts'
    - 'git grep -Ein "car|race-pack|race tick|raceState|race-step" src/neat/nge-collective/neat.nge-collective.two-population.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-collective/neat.nge-collective.two-population'
  rollback:
    - 'git checkout -- src/neat/nge-collective/neat.nge-collective.two-population.ts src/neat/nge-collective/neat.nge-collective.two-population.test.ts'
  next: 'Slice 03-05 green validation complete; proceed to slice 03-06 implementation'
  validation_evidence:
    - 'AC-03-05-001: git grep -Ein "car|race-pack|race tick|raceState|race-step" src/neat/nge-collective/neat.nge-collective.two-population.ts -> ZERO_HITS (exit 1, no matches)'
    - 'AC-03-05-002: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-collective/neat.nge-collective.two-population -> 1 suite / 10 tests pass (exit 0)'
    - 'Extra grep: git grep -Ein "createRaceStateFixture|carOrder|raceState" src/neat/nge-collective/neat.nge-collective.two-population.test.ts -> ZERO_HITS (exit 1, no matches)'
    - 'Focused coverage: src/neat/nge-collective/neat.nge-collective.two-population.ts -> 100% Stmts / 100% Branch / 100% Funcs / 100% Lines'
    - 'Broader nge-collective slice: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-collective -> 7 suites / 89 tests pass (exit 0)'
    - 'npx tsc --noEmit -p tsconfig.json: PASS (exit 0)'
    - 'npm run lint: PASS (0 issues)'
    - 'npx prettier --check on 2 changed .ts files: PASS (all use Prettier code style)'
    - 'GPU Real-Device Gate: waived (identifier-only slice, no GPU logic changes)'
    - 'plan-sync gate: pass (re-run after plan edit)'
    - 'step-packet gate: pass (re-run after plan edit)'
    - 'agent-graph gate: pass'
    - 'plan-slice-quality gate: pass'
    - 'learning-event gate: pass'
    - 'coverage-guard delegated: src/neat/nge-collective/neat.nge-collective.two-population.ts at 100% Stmts/Branch/Funcs/Lines'
    - 'cortex-index gate: FAIL (index rebuilt via node rag-index/build-index.mjs and snapshot regenerated via npm run index:build-snapshot; gate still reports snapshot_indexed_at null. Non-blocking for this rename-only slice; route to 00-helping / helping-gap-resolution-coordinator for workflow-gardening.)'
  blockers:
    - 'cortex-index gate tooling false-negative after rebuild/snapshot (not a slice validation blocker)'
```

## PlanUpdate: slice 03-06

```yaml
PlanUpdate:
  slice_id: 03-06
  changed_files:
    - src/neat.ts
    - src/neat/export/neat.export.ts
    - src/architecture/network/visualization/network.visualization.ts
    - src/architecture/network/worker-payload/network.worker-payload.browser-url.ts
    - src/visualization/network-view/network-view.types.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check src/neat.ts src/neat/export/neat.export.ts src/architecture/network/visualization/network.visualization.ts src/architecture/network/worker-payload/network.worker-payload.browser-url.ts src/visualization/network-view/network-view.types.ts'
    - 'git grep -Ein "neatchat:" src/neat.ts src/neat/export/neat.export.ts'
    - 'git grep -Ein "Flappy Bird|ASCII Maze|flappy-shared-inference" src/architecture/network/visualization/network.visualization.ts src/architecture/network/worker-payload/*.ts src/visualization/network-view/*.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/export/neat.export.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/visualization/network-view/network-view.test.ts'
  rollback:
    - 'git checkout -- src/neat.ts src/neat/export/neat.export.ts src/architecture/network/visualization/network.visualization.ts src/architecture/network/worker-payload/network.worker-payload.browser-url.ts src/visualization/network-view/network-view.types.ts'
  next: 'Dispatch 04-implementing for slice 03-07 (test fixtures and comments), then 05-green-testing'
  validation_evidence:
    - 'AC-03-06-001: git grep -Ein "neatchat:" src/neat.ts src/neat/export/neat.export.ts -> ZERO_HITS (exit 1, no matches)'
    - 'AC-03-06-001-extra: replaced prose reference "consumers such as NEATchat" with "consumers such as a downstream application" in neat.export.ts'
    - 'AC-03-06-002: git grep -Ein "Flappy Bird|ASCII Maze|flappy-shared-inference" src/architecture/network/visualization/network.visualization.ts src/architecture/network/worker-payload/*.ts src/visualization/network-view/network-view.types.ts -> ZERO_HITS (exit 1, no matches)'
    - 'AC-03-06-002-note: src/visualization/network-view/network-view.test.ts still contains a demo-specific comment; test-file sanitization is intentionally scoped to slice 03-07'
    - 'npx tsc --noEmit -p tsconfig.json: PASS (exit 0)'
    - 'npm run lint: PASS (0 issues)'
    - 'npx prettier --check on 5 changed files: PASS (all use Prettier code style)'
    - 'plan-sync gate: pass (re-run after plan edit)'
    - 'step-packet gate: pass (re-run after plan edit)'
    - 'agent-graph gate: pass'
    - 'learning-event gate: pass'
    - 'GPU Real-Device Gate: waived per task packet (JSDoc-only slice, no GPU logic changes)'
  blockers: []
```

## PlanUpdate: slice 03-06 green validation

```yaml
PlanUpdate:
  slice_id: 03-06
  role: 05-green-testing
  changed_files:
    - src/neat.ts
    - src/neat/export/neat.export.ts
    - src/architecture/network/visualization/network.visualization.ts
    - src/architecture/network/worker-payload/network.worker-payload.browser-url.ts
    - src/visualization/network-view/network-view.types.ts
  validation_run:
    - command: 'git grep -Ein "neatchat:" src/neat.ts src/neat/export/neat.export.ts'
      result: 'ZERO_HITS (exit 1, no matches)'
      gate: AC-03-06-001
      pass: true
    - command: 'git grep -Ein "Flappy Bird|ASCII Maze|flappy-shared-inference" src/architecture/network/visualization/network.visualization.ts src/architecture/network/worker-payload/*.ts src/visualization/network-view/network-view.types.ts'
      result: 'ZERO_HITS (exit 1, no matches)'
      gate: AC-03-06-002
      pass: true
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/export/neat.export'
      result: 'PASS — 4 suites, 110 tests, exit 0'
      pass: true
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/visualization/network-view/network-view'
      result: 'PASS — 1 suite, 22 tests, exit 0'
      pass: true
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/visualization/network.visualization'
      result: 'PASS — 1 suite, 33 tests, exit 0'
      pass: true
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/worker-payload'
      result: 'PASS — 5 suites, 143 tests, exit 0'
      pass: true
    - command: 'npx tsc --noEmit -p tsconfig.json'
      result: 'PASS (exit 0, 0 errors)'
      pass: true
    - command: 'npm run lint'
      result: 'PASS (0 issues)'
      pass: true
    - command: 'npx prettier --check src/neat.ts src/neat/export/neat.export.ts src/architecture/network/visualization/network.visualization.ts src/architecture/network/worker-payload/network.worker-payload.browser-url.ts src/visualization/network-view/network-view.types.ts'
      result: 'PASS (all matched files use Prettier code style)'
      pass: true
  coverage_summary:
    - file: src/neat.ts
      statements: 61.7
      branches: 79.16
      functions: 41.86
      lines: 61.42
      note: 'JSDoc-only change on a public barrel with pre-existing partial coverage; no executable code changed, no regression'
    - file: src/neat/export/neat.export.ts
      statements: 100
      branches: 100
      functions: 100
      lines: 100
    - file: src/architecture/network/visualization/network.visualization.ts
      statements: 100
      branches: 100
      functions: 100
      lines: 100
    - file: src/architecture/network/worker-payload/network.worker-payload.browser-url.ts
      statements: 100
      branches: 100
      functions: 100
      lines: 100
    - file: src/visualization/network-view/network-view.types.ts
      statements: 0
      branches: 0
      functions: 0
      lines: 0
      note: 'types-only file with no runtime executable code; JSDoc-only change, not a coverage regression'
  workflow_gates:
    - gate: plan-sync
      pass: true
      owner: validate-plan-sync.mjs
    - gate: step-packet
      pass: true
      owner: step-packet.gate.mjs
    - gate: agent-graph
      pass: true
      owner: validate-agent-graph.mjs
    - gate: learning-event
      pass: true
      owner: .github/ai-learning/learning-log.jsonl
    - gate: plan-slice-quality
      pass: true
      owner: plan-slice-quality.gate.mjs
  gpu_real_device_gate: 'waived — JSDoc-only slice, no GPU logic changes'
  blockers: []
  next: 'Slice 03-06 is green; dispatch 04-implementing for slice 03-07'
```

## PlanUpdate: slice 03-07

```yaml
PlanUpdate:
  slice_id: 03-07
  changed_files:
    - src/neat/export/neat.export.test.ts
    - src/visualization/network-view/network-view.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check src/neat/export/neat.export.test.ts src/visualization/network-view/network-view.test.ts'
    - 'git grep -Ein "racing-browser|race-step|neatchat:|createRaceStateFixture|raceState: unknown" src/**/*.test.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/export/neat.export.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/visualization/network-view/network-view.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts'
  rollback:
    - 'git checkout -- src/neat/export/neat.export.test.ts src/visualization/network-view/network-view.test.ts'
  next: 'Dispatch 05-green-testing for slice 03-07 focused validation, then proceed to Phase 4 green validation'
  validation_evidence:
    - 'AC-03-07-001: git grep -Ein "racing-browser|race-step|neatchat:|createRaceStateFixture|raceState: unknown" src/**/*.test.ts -> ZERO_HITS (exit 1, no matches)'
    - 'AC-03-07-001-extra: git grep -Ein "Flappy Bird|flappy|flappy_bird|ASCII Maze|asciiMaze|ascii_maze|racing-curriculum|racing_browser|racingBrowser|race-step|race_step|createRaceStateFixture|neatchat|neatChat|raceState|racePack|raceTick|race tick|race-pack|carOrder|carSlot|demo-specific|demo specific" src/neat/export/neat.export.test.ts src/visualization/network-view/network-view.test.ts -> ZERO_HITS (exit 1, no matches)'
    - 'Changed src/neat/export/neat.export.test.ts: replaced fixture key neatchat: with myApp: (lines 1258, 2104) to match source JSDoc examples'
    - 'Changed src/visualization/network-view/network-view.test.ts: replaced demo-specific comment referencing Flappy Bird and ASCII Maze demos with generic browser-test harness reference (line 4)'
    - 'npx tsc --noEmit -p tsconfig.json: PASS (exit 0)'
    - 'npm run lint: PASS (0 issues)'
    - 'npx prettier --check on 2 changed .test.ts files: PASS (all use Prettier code style)'
    - 'git status --porcelain: src/neat/export/neat.export.test.ts and src/visualization/network-view/network-view.test.ts modified (expected); no unintended edits'
    - 'plan-sync gate: pass'
    - 'step-packet gate: pass'
    - 'plan-slice-quality gate: pass'
    - 'agent-graph gate: pass'
    - 'learning-event gate: pass'
    - 'stale-wip-plans gate: pass'
    - 'GPU Real-Device Gate: waived per task packet (test fixture/comments only, no GPU logic changes)'
  blockers: []
```

### Validation evidence for Phase 3

## Latest validation evidence

- **Plan readiness verification:** `green-light: true` — fresh `01-planning` verification pass completed.
- `plan-sync` gate: **PASS** — `node scripts/agent-customization/gates/plan-sync.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md`
- `step-packet` gate: **PASS** — `node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md`
- `plan-slice-quality` gate: **PASS** — `node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md`
- `plan-readiness` gate: **PASS** after green-light recorded.
- **Slice 03-07 green validation (05-green-testing):** AC-03-07-001 and AC-03-07-001-extra git grep checks → ZERO_HITS; focused Jest slices for neat.export (77 pass), network-view (22 pass), evaluation-pack (8 pass), gpu.parity (11 pass), and nge-collective.two-population (10 pass) all pass; focused coverage on `src/neat/export/neat.export.ts` and `src/visualization/network-view/*.ts` is 100% Stmts/Branch/Funcs/Lines; tsc, lint, prettier pass; all Tier-1 gates pass; GPU Real-Device Gate waived; Phase 3 implementation COMPLETE.
- **Phase 2 / Step 02 research evidence:**
  - WebGPU dependency: `plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md` active slice `02-05b-buffer-parallel` is **paused** by user directive; uncommitted edits in `network.gpu.racing.ts` / `network.gpu.racing.test.ts` are from the functionally-green paused slice, not from new active work.
  - Import boundary: `git grep` across `src/` and `examples/` shows **zero** external consumers of `network.gpu.racing.ts` or the exported symbols `RacingBatchOptions`, `evaluateRacingGeneration`, `RacingAgentRequest`, `evaluateConcurrentRacingAgents` outside the co-located test.
  - Research artifact: boundary map recorded in `docs/research/public-library-demo-agnostic-import-boundary-map.md`.
  - Verdict: dependency is **CLEAR**; rename can proceed.
- **Phase 2 → Phase 3 transition gates (re-run after plan edits):**
  - `step-packet` gate: **PASS**
  - `plan-sync` gate: **PASS**
  - `plan-slice-quality` gate: **PASS**
  - Workflow update sync hook: **PASS** (`between-steps` — Phase 3 is [WIP] and Step 03 is ready for dispatch).
- **Slice 03-03 validation evidence:**
  - AC-03-03-001: `git grep -Ein "racing-curriculum|racing generation|racing eligibility|racing agents|racing-browser|demo" src/architecture/network/gpu/*.ts` → ZERO_HITS (exit 1, no matches).
  - AC-03-03-002: `git grep -Ein "racing, predator/prey, and ant-hive|RacingRenderFrame|track physics|racing-specific" src/architecture/network/evaluation-pack/*.ts` → ZERO_HITS (exit 1, no matches).
  - Fixed `src/architecture/network/gpu/network.gpu.parity.test.ts:50` — replaced "racing-browser" with "batch-evaluation browser".
  - Fixed `src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts:26` — replaced "racing-specific" with "domain-specific".
  - Fixed `src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts:125` — replaced "race-step transport" with "episode-step transport".
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.batch-evaluation` → 12/12 pass, exit 0.
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.parity` → 13/13 pass across 2 suites, exit 0.
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/evaluation-pack/network.evaluation-pack` → 8/8 pass, exit 0.
  - Focused coverage on `src/architecture/network/gpu/network.gpu.batch-evaluation.ts` → 100% Stmts / 100% Branch / 100% Funcs / 100% Lines.
  - Focused coverage on `src/architecture/network/gpu/network.gpu.batched.ts` → 100% Stmts / 100% Branch / 100% Funcs / 100% Lines.
  - Focused coverage on `src/architecture/network/evaluation-pack/network.evaluation-pack.ts` → 100% Stmts / 100% Branch / 100% Funcs / 100% Lines.
  - `code-quality-auditor` delegated: `npx tsc --noEmit -p tsconfig.json`, `npx tsc --noEmit -p tsconfig.test.json`, `npm run lint`, `npx prettier --check` on 5 slice `.ts` files → all pass, 0 issues.
  - `coverage-guard` delegated: all 3 changed source files at 100% Stmts/Branch/Funcs/Lines.
  - `npx tsc --noEmit -p tsconfig.json`: **PASS**
  - `npx tsc --noEmit -p tsconfig.test.json`: **PASS**
  - `npm run lint`: **PASS**
  - `npx prettier --check` on slice .ts files: **PASS**
  - `plan-sync` gate: **PASS** (re-run after plan edit)
  - `step-packet` gate: **PASS** (re-run after plan edit)
  - `plan-slice-quality` gate: **PASS** (re-run after plan edit)
  - `agent-graph` gate: **PASS**
  - `learning-event` gate: **PASS**
  - GPU Real-Device Gate: waived (comment-only changes in test files, no GPU logic changes; source-file JSDoc changes are doc-only).
- **Slice 03-05 validation evidence (05-green-testing):**
  - AC-03-05-001: `git grep -Ein "car|race-pack|race tick|raceState|race-step" src/neat/nge-collective/neat.nge-collective.two-population.ts` → ZERO_HITS (exit 1, no matches).
  - AC-03-05-002: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-collective/neat.nge-collective.two-population` → **1 suite / 10 tests pass** (exit 0).
  - Extra grep: `git grep -Ein "createRaceStateFixture|carOrder|raceState" src/neat/nge-collective/neat.nge-collective.two-population.test.ts` → ZERO_HITS (exit 1, no matches).
  - Focused coverage: `src/neat/nge-collective/neat.nge-collective.two-population.ts` → **100% Stmts / 100% Branch / 100% Funcs / 100% Lines**.
  - Broader nge-collective slice: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-collective` → **7 suites / 89 tests pass** (exit 0).
  - `npx tsc --noEmit -p tsconfig.json`: **PASS** (exit 0).
  - `npm run lint`: **PASS** (0 issues).
  - `npx prettier --check` on 2 changed `.ts` files: **PASS** (all use Prettier code style).
  - **Tier-1 gates (re-run after plan edit):** `plan-sync` PASS, `agent-graph` PASS, `step-packet` PASS, `plan-slice-quality` PASS, `learning-event` PASS.
  - **GPU Real-Device Gate:** waived (identifier-only slice, no GPU logic changes).
  - **Delegated `coverage-guard`:** `src/neat/nge-collective/neat.nge-collective.two-population.ts` at 100% Stmts/Branch/Funcs/Lines.
  - **Workflow gap:** `cortex-index` gate reports FAIL after `node rag-index/build-index.mjs` and `npm run index:build-snapshot` (snapshot_indexed_at still null). This is a repo-wide index tooling issue, not a slice blocker; route to `00-helping` / `helping-gap-resolution-coordinator` for gardening.
- **Slice 03-05 validation evidence (04-implementing preflight):**
  - AC-03-05-001: `git grep -Ein "car|race-pack|race tick|raceState|race-step" src/neat/nge-collective/neat.nge-collective.two-population.ts` → ZERO_HITS (exit 1, no matches).
  - AC-03-05-002 (pending 05-green-testing): `npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-collective/neat.nge-collective.two-population`.
  - `npx tsc --noEmit -p tsconfig.json`: **PASS** (exit 0).
  - `npm run lint`: **PASS** (0 issues).
  - `npx prettier --check` on 2 changed `.ts` files: **PASS** (all use Prettier code style).
  - **Tier-1 gates:** `plan-sync` PASS, `agent-graph` PASS, `step-packet` PASS.
  - **GPU Real-Device Gate:** waived (identifier-only slice, no GPU logic changes).
- **Slice 03-06 validation evidence (04-implementing preflight):**
  - AC-03-06-001: `git grep -Ein "neatchat:" src/neat.ts src/neat/export/neat.export.ts` → ZERO_HITS (exit 1, no matches).
  - AC-03-06-001-extra: replaced prose reference "consumers such as NEATchat" with "consumers such as a downstream application" in `src/neat/export/neat.export.ts`.
  - AC-03-06-002: `git grep -Ein "Flappy Bird|ASCII Maze|flappy-shared-inference" src/architecture/network/visualization/network.visualization.ts src/architecture/network/worker-payload/*.ts src/visualization/network-view/network-view.types.ts` → ZERO_HITS (exit 1, no matches).
  - AC-03-06-002-note: `src/visualization/network-view/network-view.test.ts` still contains a demo-specific comment; test-file sanitization is intentionally scoped to slice 03-07.
  - `npx tsc --noEmit -p tsconfig.json`: **PASS** (exit 0).
  - `npm run lint`: **PASS** (0 issues).
  - `npx prettier --check` on 5 changed `.ts` files: **PASS** (all use Prettier code style).
  - **Tier-1 gates:** `plan-sync` PASS, `step-packet` PASS, `agent-graph` PASS, `learning-event` PASS.
  - **GPU Real-Device Gate:** waived per task packet (JSDoc-only slice, no GPU logic changes).
- **Slice 03-06 green validation (05-green-testing):**
  - AC-03-06-001: `git grep -Ein "neatchat:" src/neat.ts src/neat/export/neat.export.ts` → ZERO_HITS (exit 1, no matches).
  - AC-03-06-002: `git grep -Ein "Flappy Bird|ASCII Maze|flappy-shared-inference" src/architecture/network/visualization/network.visualization.ts src/architecture/network/worker-payload/*.ts src/visualization/network-view/network-view.types.ts` → ZERO_HITS (exit 1, no matches).
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/export/neat.export` → **4 suites / 110 tests pass** (exit 0).
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/visualization/network-view/network-view` → **1 suite / 22 tests pass** (exit 0).
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/visualization/network.visualization` → **1 suite / 33 tests pass** (exit 0).
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/worker-payload` → **5 suites / 143 tests pass** (exit 0).
  - Focused coverage: `src/neat/export/neat.export.ts`, `src/architecture/network/visualization/network.visualization.ts`, and `src/architecture/network/worker-payload/network.worker-payload.browser-url.ts` all at **100% Stmts / 100% Branch / 100% Funcs / 100% Lines**; `src/neat.ts` JSDoc-only with no executable-code change and no regression; `src/visualization/network-view/network-view.types.ts` is types-only.
  - `npx tsc --noEmit -p tsconfig.json`: **PASS** (exit 0, 0 errors).
  - `npm run lint`: **PASS** (0 issues).
  - `npx prettier --check` on 5 changed `.ts` files: **PASS** (all use Prettier code style).
  - **Tier-1 gates (re-run after plan edit):** `plan-sync` PASS, `step-packet` PASS, `agent-graph` PASS, `learning-event` PASS, `plan-slice-quality` PASS.
  - **GPU Real-Device Gate:** waived per task packet (JSDoc-only slice, no GPU logic changes).
- **Slice 03-07 validation evidence (04-implementing preflight):**
  - AC-03-07-001: `git grep -Ein "racing-browser|race-step|neatchat:|createRaceStateFixture|raceState: unknown" src/**/*.test.ts` → ZERO_HITS (exit 1, no matches).
  - AC-03-07-001-extra: `git grep -Ein "Flappy Bird|flappy|flappy_bird|ASCII Maze|asciiMaze|ascii_maze|racing-curriculum|racing_browser|racingBrowser|race-step|race_step|createRaceStateFixture|neatchat|neatChat|raceState|racePack|raceTick|race tick|race-pack|carOrder|carSlot|demo-specific|demo specific" src/neat/export/neat.export.test.ts src/visualization/network-view/network-view.test.ts` → ZERO_HITS (exit 1, no matches).
  - Replaced fixture key `neatchat:` with `myApp:` in `src/neat/export/neat.export.test.ts` lines 1258 and 2104 to match source JSDoc examples.
  - Replaced demo-specific comment referencing Flappy Bird and ASCII Maze demos with generic browser-test harness reference in `src/visualization/network-view/network-view.test.ts` line 4.
  - `npx tsc --noEmit -p tsconfig.json`: **PASS** (exit 0).
  - `npm run lint`: **PASS** (0 issues).
  - `npx prettier --check` on 2 changed `.test.ts` files plus the plan file: **PASS** (all use Prettier code style).
  - `git status --porcelain`: only expected test files modified; no unintended edits.
  - **Tier-1 gates (re-run after plan edit):** `plan-sync` PASS, `step-packet` PASS, `plan-slice-quality` PASS, `agent-graph` PASS, `learning-event` PASS, `stale-wip-plans` PASS.
  - **GPU Real-Device Gate:** waived per task packet (test fixture/comments only, no GPU logic changes).
- **Slice 03-07 green validation (05-green-testing):**
  - AC-03-07-001: `git grep -Ein "racing-browser|race-step|neatchat:|createRaceStateFixture|raceState: unknown" src/**/*.test.ts` → **ZERO_HITS** (exit 1, no matches).
  - AC-03-07-001-extra: `git grep -Ein "Flappy Bird|flappy|flappy_bird|ASCII Maze|asciiMaze|ascii_maze|racing-curriculum|racing_browser|racingBrowser|race-step|race_step|createRaceStateFixture|neatchat|neatChat|raceState|racePack|raceTick|race tick|race-pack|carOrder|carSlot|demo-specific|demo specific" src/neat/export/neat.export.test.ts src/visualization/network-view/network-view.test.ts` → **ZERO_HITS** (exit 1, no matches).
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/export/neat.export.test.ts` → **1 suite / 77 tests pass** (exit 0).
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/visualization/network-view/network-view.test.ts` → **1 suite / 22 tests pass** (exit 0).
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts` → **1 suite / 8 tests pass** (exit 0).
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.parity.test.ts` → **1 suite / 11 tests pass** (exit 0).
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-collective/neat.nge-collective.two-population.test.ts` → **1 suite / 10 tests pass** (exit 0).
  - Focused coverage: `src/neat/export/neat.export.ts` → **100% Stmts / 100% Branch / 100% Funcs / 100% Lines**.
  - Focused coverage: `src/visualization/network-view/network-view.ts`, `src/visualization/network-view/network-view.layout.utils.ts`, `src/visualization/network-view/network-view.topology.utils.ts` → **100% Stmts / 100% Branch / 100% Funcs / 100% Lines**.
  - `npx tsc --noEmit -p tsconfig.json`: **PASS** (exit 0).
  - `npm run lint`: **PASS** (0 issues).
  - `npx prettier --check` on 2 changed `.test.ts` files plus the plan file: **PASS** (all use Prettier code style).
  - `git status --porcelain`: only expected test files modified; no unintended edits.
  - **Tier-1 gates (re-run after plan edit):** `plan-sync` PASS, `step-packet` PASS, `plan-slice-quality` PASS, `agent-graph` PASS, `learning-event` PASS, `stale-wip-plans` PASS.
  - **GPU Real-Device Gate:** waived per task packet (test fixture/comments only, no GPU logic changes).
  - **Verdict:** Slice 03-07 is green; Phase 3 implementation is COMPLETE.
- **Slice 03-01 validation evidence:**
  - `git mv src/architecture/network/gpu/network.gpu.racing.ts src/architecture/network/gpu/network.gpu.batch-evaluation.ts` — rename tracked.
  - `git mv src/architecture/network/gpu/network.gpu.racing.test.ts src/architecture/network/gpu/network.gpu.batch-evaluation.test.ts` — rename tracked.
  - Updated test import path from `'./network.gpu.racing'` to `'./network.gpu.batch-evaluation'`.
  - Updated `src/architecture/network/gpu/docs.order.json` to reference `network.gpu.batch-evaluation.ts`.
  - `npx tsc --noEmit -p tsconfig.json`: **PASS** (exit 0).
  - `npm run lint`: **PASS** (exit 0).
  - `npx prettier --check` on changed files: **PASS** (exit 0).
  - AC-03-01-001: `git ls-files src/architecture/network/gpu/network.gpu.racing.ts` returns empty; `git ls-files src/architecture/network/gpu/network.gpu.batch-evaluation.ts` present.
  - AC-03-01-002: `git ls-files src/architecture/network/gpu/*.test.ts | grep -i racing` returns empty.
  - AC-03-01-003: `docs.order.json` `fileOrder` includes `network.gpu.batch-evaluation.ts`.
  - **05-green-testing focused test run:** `network.gpu.batch-evaluation.test.ts` — 12/12 pass.
  - **05-green-testing regression check:** `network.gpu.batched.test.ts` — 27/27 pass.
  - **05-green-testing coverage:** `network.gpu.batch-evaluation.ts` — 100% Stmts / 100% Branch / 100% Funcs / 100% Lines.
  - **Tier-1 gates:** `plan-sync` PASS, `agent-graph` PASS, `step-packet` PASS.
  - **GPU Real-Device Gate:** waived per task packet (pure file rename, exported symbols unchanged for this slice).
  - **Delegated specialists:** `coverage-guard` (100% coverage confirmed), `code-quality-auditor` (tsc/lint/prettier confirmed).

```yaml
verification:
  agent: 01-planning
  mode: verification
  timestamp: 2026-07-04T09:33:00-04:00
  green-light: true
  blockers: []
  observations:
    - Slice 03-04 estimate_hours is exactly 4 (hard limit) and touches 9 NGE files; monitor for scope creep during implementation.
    - Phase 5 uses tdd_sequence: red-green, which is atypical for documentation-only slices but does not violate schema.
```

```json
[
  {
    "gate": "plan-sync",
    "command": "node scripts/agent-customization/gates/plan-sync.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md",
    "result": {
      "pass": true,
      "evidence": {
        "wipPlans": [
          "plans/mcp-active-binding.plans.md",
          "plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md",
          "plans/Public_Library_Demo_Agnostic_Refactor.plans.md"
        ],
        "missingFromReadme": [],
        "missingFromRoadmap": [],
        "plansChecked": 7
      },
      "fixHint": "All WIP plans are correctly registered in README and Roadmap.",
      "owner": "validate-plan-sync.mjs"
    }
  },
  {
    "gate": "step-packet",
    "command": "node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md",
    "result": {
      "pass": true,
      "evidence": {
        "blocksChecked": [
          "plans/mcp-active-binding.plans.md:yaml@3934",
          "plans/mcp-active-binding.plans.md:yaml@5387",
          "plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md:yaml@186191",
          "plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md:yaml@205595",
          "plans/Public_Library_Demo_Agnostic_Refactor.plans.md:yaml@10865",
          "plans/Public_Library_Demo_Agnostic_Refactor.plans.md:yaml@11981"
        ],
        "violations": [],
        "planReadinessWarnings": [],
        "plansScanned": 4
      },
      "fixHint": "All active WIP phase/step packets conform to the new format.",
      "owner": "step-packet.gate.mjs"
    }
  },
  {
    "gate": "plan-slice-quality",
    "command": "node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md",
    "result": {
      "pass": true,
      "evidence": {
        "plansChecked": [
          "plans/mcp-active-binding.plans.md",
          "plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md",
          "plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md",
          "plans/Public_Library_Demo_Agnostic_Refactor.plans.md"
        ],
        "violations": [],
        "limit": 4
      },
      "fixHint": "All WIP plan slices are within the 4-hour estimate limit.",
      "owner": "plan-slice-quality.gate.mjs"
    }
  },
  {
    "gate": "slice-green-03-01",
    "command": "npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.batch-evaluation --no-coverage; npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.batched --no-coverage; npx jest --config=jest.config.mjs --no-cache --coverage --collectCoverageFrom='src/architecture/network/gpu/network.gpu.batch-evaluation.ts' --testPathPatterns=src/architecture/network/gpu/network.gpu.batch-evaluation",
    "result": {
      "pass": true,
      "slice_id": "03-01",
      "evidence": {
        "coverage_summary": {
          "statements": 100,
          "branches": 100,
          "functions": 100,
          "lines": 100
        },
        "test_results": "network.gpu.batch-evaluation.test.ts 12/12 pass; network.gpu.batched.test.ts 27/27 pass; all exits 0",
        "gpu_real_device_gate": "waived per task packet — pure file rename, exported symbols unchanged for slice 03-01"
      },
      "fixHint": "If tests fail, verify import paths in the renamed test file and docs.order.json fileOrder.",
      "owner": "05-green-testing"
    }
  },
  {
    "gate": "slice-green-03-02",
    "command": "git grep -Ein \"RacingBatchOptions|evaluateRacingGeneration|RacingAgentRequest|evaluateConcurrentRacingAgents\" src/; git grep -Ein \"evaluateRacingGeneration\" src/architecture/network/gpu/network.gpu.batched.ts; npx jest --config=jest.config.mjs --no-cache --testPathPatterns=network.gpu.batch-evaluation --no-coverage; npx jest --config=jest.config.mjs --no-cache --testPathPatterns=network.gpu.batched --no-coverage; npx jest --config=jest.config.mjs --no-cache --testPathPatterns=network.gpu.batch-evaluation --coverage --collectCoverageFrom='src/architecture/network/gpu/network.gpu.batch-evaluation.ts'",
    "result": {
      "pass": true,
      "slice_id": "03-02",
      "evidence": {
        "coverage_summary": {
          "statements": 100,
          "branches": 100,
          "functions": 100,
          "lines": 100
        },
        "test_results": "network.gpu.batch-evaluation.test.ts 12/12 pass; network.gpu.batched.test.ts 27/27 pass; git grep for old Racing* symbols returns ZERO_HITS; all exits 0",
        "gpu_real_device_gate": "waived per task packet (symbol rename only, no logic changes, no GPU runtime change)"
      },
      "fixHint": "If old Racing* symbols remain, rerun git grep and update remaining call sites. If tests fail, verify the renamed imports in batch-evaluation.test.ts and the JSDoc reference in batched.ts.",
      "owner": "05-green-testing"
    }
  },
  {
    "gate": "step-packet-post-edit",
    "command": "node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md",
    "result": {
      "pass": true,
      "evidence": {
        "blocksChecked": [
          "plans/Public_Library_Demo_Agnostic_Refactor.plans.md:yaml@27390",
          "plans/Public_Library_Demo_Agnostic_Refactor.plans.md:yaml@28466"
        ],
        "violations": [],
        "planReadinessWarnings": []
      },
      "fixHint": "n/a",
      "owner": "step-packet.gate.mjs"
    }
  },
  {
    "gate": "plan-sync-post-edit",
    "command": "node scripts/agent-customization/gates/plan-sync.gate.mjs --json",
    "result": {
      "pass": true,
      "evidence": {
        "wipPlans": ["plans/Public_Library_Demo_Agnostic_Refactor.plans.md"],
        "missingFromReadme": [],
        "missingFromRoadmap": []
      },
      "fixHint": "n/a",
      "owner": "validate-plan-sync.mjs"
    }
  },
  {
    "gate": "slice-green-03-04",
    "command": "git grep -Ein \"(racing|ant-hive|predator/prey|racing-worker)\" src/neat/nge-*.ts src/neat/nge-*/*.ts src/neat/neat.nge-lifecycle.ts; git grep -Ein \"Racing|Ant Hive\" src/neat/nge-collective/*.ts; npx tsc --noEmit -p tsconfig.json; npm run lint; npx prettier --check src/neat/neat.nge-lifecycle.ts src/neat/nge-collective/neat.nge-collective.shared-field.ts src/neat/nge-collective/neat.nge-collective.team-fitness.ts src/neat/nge-collective/neat.nge-collective.ts src/neat/nge-collective/neat.nge-collective.types.ts src/neat/nge-dna/neat.nge-dna.ts src/neat/nge-dna/neat.nge-dna.types.ts src/neat/nge-evolution/neat.nge-evolution.reproduction.ts src/neat/nge-juvenile/neat.nge-juvenile.ts",
    "result": {
      "pass": true,
      "slice_id": "03-04",
      "evidence": {
        "grep_racing": "ZERO_HITS in slice files; only non-slice src/neat/nge-collective/neat.nge-collective.two-population.ts still contains 'racing' (slice 03-05)",
        "grep_racing_ant_hive": "ZERO_HITS in slice files; only non-slice src/neat/nge-collective/neat.nge-collective.two-population.ts still contains 'racing' (slice 03-05)",
        "tsc": "PASS",
        "lint": "PASS (0 issues)",
        "prettier": "PASS (all 9 changed files use Prettier code style)",
        "gpu_real_device_gate": "waived per task packet — JSDoc-only slice, no GPU logic changes"
      },
      "fixHint": "If grep finds demo-specific terms in slice files, replace them with generic library-appropriate language. If tsc/lint/prettier fail, fix only JSDoc comments and whitespace.",
      "owner": "04-implementing"
    }
  },
  {
    "gate": "plan-sync-post-03-04",
    "command": "node scripts/agent-customization/gates/plan-sync.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md",
    "result": {
      "pass": true,
      "evidence": {
        "wipPlans": [
          "plans/mcp-active-binding.plans.md",
          "plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md",
          "plans/Public_Library_Demo_Agnostic_Refactor.plans.md"
        ],
        "missingFromReadme": [],
        "missingFromRoadmap": [],
        "plansChecked": 7
      },
      "fixHint": "All WIP plans are correctly registered in README and Roadmap.",
      "owner": "validate-plan-sync.mjs"
    }
  },
  {
    "gate": "step-packet-post-03-04",
    "command": "node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md",
    "result": {
      "pass": true,
      "evidence": {
        "blocksChecked": [
          "plans/mcp-active-binding.plans.md:yaml@3934",
          "plans/mcp-active-binding.plans.md:yaml@5387",
          "plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md:yaml@186275",
          "plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md:yaml@205679",
          "plans/Public_Library_Demo_Agnostic_Refactor.plans.md:yaml@39256",
          "plans/Public_Library_Demo_Agnostic_Refactor.plans.md:yaml@40332"
        ],
        "violations": [],
        "planReadinessWarnings": [],
        "plansScanned": 4
      },
      "fixHint": "All active WIP phase/step packets conform to the new format.",
      "owner": "step-packet.gate.mjs"
    }
  },
  {
    "gate": "plan-slice-quality-post-03-04",
    "command": "node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md",
    "result": {
      "pass": true,
      "evidence": {
        "plansChecked": [
          "plans/mcp-active-binding.plans.md",
          "plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md",
          "plans/NEAT_Genesis_EvoDevo_WebGPU_Real_Performance.plans.md",
          "plans/Public_Library_Demo_Agnostic_Refactor.plans.md"
        ],
        "violations": [],
        "limit": 4
      },
      "fixHint": "All WIP plan slices are within the 4-hour estimate limit.",
      "owner": "plan-slice-quality.gate.mjs"
    }
  },
  {
    "gate": "agent-graph",
    "command": "node scripts/agent-customization/gates/agent-graph.gate.mjs --json",
    "result": {
      "pass": true,
      "evidence": {
        "ok": true,
        "issueCount": 0,
        "agentCount": 67,
        "byTier": { "1": 8, "2": 11, "3": 44, "4": 4 },
        "issues": []
      },
      "fixHint": "Agent delegation graph is valid; references resolve, no cycles exist, and tier enforcement rules pass.",
      "owner": "validate-agent-graph.mjs"
    }
  },
  {
    "gate": "learning-event",
    "command": "node scripts/agent-customization/gates/learning-event.gate.mjs --json",
    "result": {
      "pass": true,
      "evidence": {
        "exists": true,
        "path": ".github/ai-learning/learning-log.jsonl",
        "eventCount": 16583,
        "rawLineCount": 16584,
        "categories": [
          "runtime-action-prepass",
          "runtime-action-postpass",
          "runtime-proof-mismatch",
          "gate-exception",
          "gate-escalation",
          "agent-system-gap",
          "output-contract-fix",
          "skill-update",
          "workflow-gap-remediation",
          "workflow-improvement",
          "agent-update",
          "workflow-recovery",
          "tier-delegation-fix",
          "constitution-update"
        ]
      },
      "fixHint": "Learning event log exists and contains at least one valid event.",
      "owner": ".github/ai-learning/learning-log.jsonl"
    }
  }
]
```

---

### Step 03 packet

### Phase 3 — Implementation [DONE]

```yaml
phase: 3
title: Implementation
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/Public_Library_Demo_Agnostic_Refactor.plans.md
copy_paste: true
next_phase: 'Green Validation'
active_step: 'Step 03 — Rename critical GPU file, symbols, and demo-specific JSDoc'
skills:
  - plan-alignment
validation:
  - 'node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md'
acceptance_criteria:
  - 'Plan phase/step YAML blocks pass the step-packet gate.'
placeholder_steps:
  - 'Step 03 — Rename critical GPU file, symbols, and demo-specific JSDoc'
```

**Phase objective:** Execute the demo-agnostic refactor in `src/` using green-only TDD (tests already exist and will be co-renamed).

**Phase progression rule:** Implementation is sliced by severity and subsystem. Each slice removes old names in the same pass that introduces new names.

#### Step 03: Rename critical GPU file, symbols, and demo-specific JSDoc [DONE]

```yaml
phase: 3
step: 3
title: 'Rename critical GPU file, symbols, and demo-specific JSDoc'
status: '[DONE]'
goal: implementing
tdd_sequence: green-only
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/Public_Library_Demo_Agnostic_Refactor.plans.md
copy_paste: true
next_step: 'Step 04 — Green validation'
active_slice: 03-07
skills:
  - implementation-standards
validation:
  - 'npx tsc --noEmit -p tsconfig.json'
  - 'npm run lint'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=network.gpu.batch-evaluation'
acceptance_criteria:
  - 'GPU batch-evaluation module and test file are renamed and old racing file is removed'
  - 'Network.ts no longer references racing-named GPU symbols'
  - 'All demo-specific JSDoc and comments in touched src/ files are sanitized'
  - 'All demo-specific internal identifiers in touched src/ files are renamed to generic terms'
  - 'Co-renamed tests pass and 100% coverage is maintained on touched src/ files'
  - 'TypeScript compilation and lint pass across src/'
  - 'No examples/ or docs/browser-tests/ files are changed'
specialists:
  - implementation-executor
  - solid-split
slices:
  - slice_id: 03-01
    title: 'Rename GPU batch-evaluation module and test file'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - src/architecture/network/gpu/network.gpu.racing.ts
      - src/architecture/network/gpu/network.gpu.racing.test.ts
      - src/architecture/network/gpu/network.gpu.batch-evaluation.ts
      - src/architecture/network/gpu/network.gpu.batch-evaluation.test.ts
      - src/architecture/network/gpu/docs.order.json
    acceptance_criteria:
      - id: AC-03-01-001
        text: 'network.gpu.racing.ts is renamed to network.gpu.batch-evaluation.ts and the old path is removed'
        validation: 'git ls-files src/architecture/network/gpu/network.gpu.racing.ts; git ls-files src/architecture/network/gpu/network.gpu.batch-evaluation.ts'
      - id: AC-03-01-002
        text: 'network.gpu.racing.test.ts is renamed to network.gpu.batch-evaluation.test.ts'
        validation: 'git ls-files src/architecture/network/gpu/*.test.ts | grep -i racing'
      - id: AC-03-01-003
        text: 'docs.order.json references the new file name'
        validation: 'node -e "const o=require(\''./src/architecture/network/gpu/docs.order.json\''); console.log(JSON.stringify(o.fileOrder))"'
    parallelizable: false
    dependencies:
    next_slice: 03-02
  - slice_id: 03-02
    title: 'Rename exported Racing* symbols and update internal callers'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - src/architecture/network/gpu/network.gpu.batch-evaluation.ts
      - src/architecture/network/gpu/network.gpu.batch-evaluation.test.ts
      - src/architecture/network/gpu/network.gpu.batched.ts
    acceptance_criteria:
      - id: AC-03-02-001
        text: 'RacingBatchOptions -> BatchEvaluationOptions, evaluateRacingGeneration -> evaluateBatchGeneration, RacingAgentRequest -> AgentEvaluationRequest, evaluateConcurrentRacingAgents -> evaluateConcurrentAgents'
        validation: 'git grep -Ein "RacingBatchOptions|evaluateRacingGeneration|RacingAgentRequest|evaluateConcurrentRacingAgents" src/'
      - id: AC-03-02-002
        text: 'network.gpu.batched.ts JSDoc references the renamed evaluateBatchGeneration'
        validation: 'git grep -Ein "evaluateRacingGeneration" src/architecture/network/gpu/network.gpu.batched.ts'
      - id: AC-03-02-003
        text: 'batch-evaluation tests pass after symbol rename'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=network.gpu.batch-evaluation'
    parallelizable: false
    dependencies:
      - 03-01
    next_slice: 03-03
  - slice_id: 03-03
    title: 'Sanitize GPU and evaluation-pack JSDoc'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - src/architecture/network/gpu/network.gpu.batch-evaluation.ts
      - src/architecture/network/gpu/network.gpu.batched.ts
      - src/architecture/network/gpu/network.gpu.parity.test.ts
      - src/architecture/network/evaluation-pack/network.evaluation-pack.ts
      - src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts
    acceptance_criteria:
      - id: AC-03-03-001
        text: 'No racing-curriculum / demo references remain in GPU module JSDoc'
        validation: 'git grep -Ein "racing-curriculum|racing generation|racing eligibility|racing agents|racing-browser|demo" src/architecture/network/gpu/*.ts'
      - id: AC-03-03-002
        text: 'Evaluation-pack JSDoc and Mermaid diagrams use generic benchmark language'
        validation: 'git grep -Ein "racing, predator/prey, and ant-hive|RacingRenderFrame|track physics|racing-specific" src/architecture/network/evaluation-pack/*.ts'
    parallelizable: false
    dependencies:
      - 03-02
    next_slice: 03-04
  - slice_id: 03-04
    title: 'Sanitize NGE JSDoc and public examples'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 4
    files_to_change:
      - src/neat/neat.nge-lifecycle.ts
      - src/neat/nge-collective/neat.nge-collective.shared-field.ts
      - src/neat/nge-collective/neat.nge-collective.team-fitness.ts
      - src/neat/nge-collective/neat.nge-collective.ts
      - src/neat/nge-collective/neat.nge-collective.types.ts
      - src/neat/nge-dna/neat.nge-dna.ts
      - src/neat/nge-dna/neat.nge-dna.types.ts
      - src/neat/nge-evolution/neat.nge-evolution.reproduction.ts
      - src/neat/nge-juvenile/neat.nge-juvenile.ts
    acceptance_criteria:
      - id: AC-03-04-001
        text: 'NGE module JSDoc no longer names racing, ant-hive, or predator/prey demos'
        validation: 'git grep -Ein "(racing|ant-hive|predator/prey|racing-worker)" src/neat/nge-*.ts src/neat/nge-*/*.ts src/neat/neat.nge-lifecycle.ts'
      - id: AC-03-04-002
        text: 'Mermaid diagrams in team-fitness and collective modules use generic consumer labels'
        validation: 'git grep -Ein "Racing|Ant Hive" src/neat/nge-collective/*.ts'
    parallelizable: false
    dependencies:
      - 03-03
    next_slice: 03-05
  - slice_id: 03-05
    title: 'Rename internal identifiers in two-population scaffold'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - src/neat/nge-collective/neat.nge-collective.two-population.ts
      - src/neat/nge-collective/neat.nge-collective.two-population.test.ts
    acceptance_criteria:
      - id: AC-03-05-001
        text: 'car/cars, race-pack, race tick, raceState, race-step are replaced by agent/slot, episode-pack, episode tick, episodeState, episode-step'
        validation: 'git grep -Ein "car|race-pack|race tick|raceState|race-step" src/neat/nge-collective/neat.nge-collective.two-population.ts'
      - id: AC-03-05-002
        text: 'Two-population tests pass after identifier rename'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-collective/neat.nge-collective.two-population'
    parallelizable: false
    dependencies:
      - 03-04
    next_slice: 03-06
  - slice_id: 03-06
    title: 'Sanitize export, neat.ts, visualization, and worker-payload JSDoc'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - src/neat.ts
      - src/neat/export/neat.export.ts
      - src/architecture/network/visualization/network.visualization.ts
      - src/architecture/network/worker-payload/network.worker-payload.browser-url.ts
      - src/visualization/network-view/network-view.types.ts
    acceptance_criteria:
      - id: AC-03-06-001
        text: 'neat.ts and neat.export.ts examples use generic consumer/myApp namespace instead of neatchat'
        validation: 'git grep -Ein "neatchat:" src/neat.ts src/neat/export/neat.export.ts'
      - id: AC-03-06-002
        text: 'Visualization and worker-payload JSDoc no longer names Flappy Bird or ASCII Maze'
        validation: 'git grep -Ein "Flappy Bird|ASCII Maze|flappy-shared-inference" src/architecture/network/visualization/network.visualization.ts src/architecture/network/worker-payload/*.ts src/visualization/network-view/*.ts'
    parallelizable: false
    dependencies:
      - 03-05
    next_slice: 03-07
  - slice_id: 03-07
    title: 'Update test fixtures and comments across affected src/ tests'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 3
    files_to_change:
      - src/architecture/network/gpu/network.gpu.parity.test.ts
      - src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts
      - src/neat/export/neat.export.test.ts
      - src/visualization/network-view/network-view.test.ts
    acceptance_criteria:
      - id: AC-03-07-001
        text: 'Test describe blocks and comments no longer use racing-browser, race-step, neatchat:, or createRaceStateFixture'
        validation: 'git grep -Ein "racing-browser|race-step|neatchat:|createRaceStateFixture|raceState: unknown" src/**/*.test.ts'
      - id: AC-03-07-002
        text: 'All affected tests compile and pass after fixture updates'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/export/neat.export.test.ts; npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/visualization/network-view/network-view.test.ts; npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts'
    parallelizable: false
    dependencies:
      - 03-06
    next_slice: null
```

**User instruction:** Paste this full step packet.

**Step objective:** Remove all demo-specific names from the public library's file names, exported symbols, JSDoc, internal identifiers, and tests in a single coordinated green-only pass.

**Context the agent must know:**

- Tests already exist; this is `tdd_sequence: green-only`.
- No backward-compatibility wrappers or dual-path code.
- Old file names and symbols must be removed in the same slice that introduces replacements.

**Execution steps:**

1. Start with slice `03-01` (file/test rename + docs.order.json).
2. Proceed sequentially through `03-02` to `03-07`.
3. After each slice, run the slice-specific validation and `npm run lint` + `npx tsc --noEmit -p tsconfig.json`.
4. Do not run `npm run docs` yet; that is Phase 5.

**Stop conditions:**

- Done when all 7 slices pass and `git grep` for old demo names in `src/**/*.ts` returns zero matches.
- Blocked if WebGPU plan re-enters the racing file; stop and return to Step 02.

**Required validation:**

- `npx tsc --noEmit -p tsconfig.json`
- `npm run lint`
- Targeted Jest commands per slice

**Plan update requirement:** Mark each slice `[DONE]` as it passes and record validation evidence. Advance to Step 04 only when all slices are green.

---


## Phase 4 � Green Validation [DONE]

[DONE] Phase 4: focused green validation passed � tsc, lint, targeted Jest slices, folder-quality gates, and 100% coverage on all touched `src/` files. Full step packet and validation evidence preserved below.

---

### Phase 4 � Green Validation [DONE]

```yaml
phase: 4
title: 'Green Validation'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/Public_Library_Demo_Agnostic_Refactor.plans.md
copy_paste: true
next_phase: Documentation
skills:
  - plan-alignment
validation:
  - 'node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md'
acceptance_criteria:
  - 'Plan phase/step YAML blocks pass the step-packet gate.'
placeholder_steps:
  - 'Step 04 � Run focused green validation'
```

**Phase objective:** Confirm the refactor compiles, lints, and passes targeted tests for every affected folder.

#### Step 04: Run focused green validation [DONE]

```yaml
phase: 4
step: 4
title: 'Run focused green validation'
status: '[DONE]'
goal: green-testing
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/Public_Library_Demo_Agnostic_Refactor.plans.md
copy_paste: true
next_step: 'Step 05 � Regenerate docs and write migration note'
skills:
  - green-validation-gates
  - coverage-guard
validation:
  - 'npx tsc --noEmit -p tsconfig.json'
  - 'npm run lint'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/gpu/network.gpu.batch-evaluation'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-collective/neat.nge-collective.two-population'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/export/neat.export.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/visualization/network-view/network-view.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts'
  - 'npm run quality:folder -- --folder=src/architecture/network/gpu'
  - 'npm run quality:folder -- --folder=src/neat/nge-collective'
acceptance_criteria:
  - 'Renamed GPU module focused test suite passes'
  - 'Full src/ test suite passes'
  - 'npm run lint exits with code 0'
  - 'npx tsc --noEmit -p tsconfig.json passes'
  - '100% statement/branch/function/line coverage on touched src/ files'
```

**User instruction:** Paste this full step packet.

**Step objective:** Validate that the renamed code, sanitized JSDoc, and updated tests are green across compilation, lint, focused Jest, folder quality, and coverage.

**Context the agent must know:**

- This is a green-only validation phase; no code edits except to fix regressions found by tests.
- If a test fails, route back to a new `04-implementing` slice-fix instance, then re-run green validation.

**Execution steps:**

1. Run `npx tsc --noEmit -p tsconfig.json`.
2. Run `npm run lint`.
3. Run each targeted Jest command in its own shell invocation.
4. Run `npm run quality:folder` for affected folders.
5. Run coverage guard on touched src/ files.

**Stop conditions:**

- Done when all validation commands pass.
- Blocked by any failure; route back to `04-implementing` with observations.

**Required validation:**

- `npx tsc --noEmit -p tsconfig.json`
- `npm run lint`
- Targeted Jest commands listed above
- `npm run quality:folder` commands

**Plan update requirement:** Record all command outputs, attach coverage report, and advance to Step 05 on green.

---

## Phase 5 � Documentation [DONE]

[DONE] Phase 5: docs regenerated via `npm run docs`, generated READMEs are demo-free, migration note added to `RELEASE.md`, final typecheck/lint/prettier passed. Full step packet and validation evidence preserved below.

---

### Phase 5 � Documentation [DONE]

```yaml
phase: 5
title: Documentation
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/Public_Library_Demo_Agnostic_Refactor.plans.md
copy_paste: true
next_phase: Logging
skills:
  - plan-alignment
validation:
  - 'node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/Public_Library_Demo_Agnostic_Refactor.plans.md'
acceptance_criteria:
  - 'Plan phase/step YAML blocks pass the step-packet gate.'
placeholder_steps:
  - 'Step 05 � Regenerate docs and write migration note'
```

**Phase objective:** Regenerate generated docs from sanitized source and write the breaking-change migration note.

#### Step 05: Regenerate docs and write migration note [DONE]

```yaml
phase: 5
step: 5
title: 'Regenerate docs and write migration note'
status: '[DONE]'
goal: documenting
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/Public_Library_Demo_Agnostic_Refactor.plans.md
copy_paste: true
next_step: 'Step 06 � Compress phase history and close'
skills:
  - educational-docs
validation:
  - 'npm run docs'
  - 'git grep -Ein "(flappy|flappy_bird|flappy-bird|asciiMaze|ASCII[ _-]?Maze|neatchat|racing_curriculum|racing-curriculum|ant[ _-]?hive|predator/prey|RacingBatchOptions|evaluateRacingGeneration|network\.gpu\.racing)" src/**/README.md'
acceptance_criteria:
  - 'npm run docs regenerates src/**/README.md files with no demo-specific names'
  - 'RELEASE.md contains a migration note mapping every removed public symbol to its replacement'
  - 'src/architecture/network/gpu/docs.order.json references the renamed batch-evaluation module'
specialists:
  - docs-example-writer
  - docs-scout
slices:
  - slice_id: 05-01
    title: 'Regenerate generated READMEs via npm run docs'
    status: '[PLANNED]'
    goal: documenting
    estimate_hours: 2
    files_to_change:
      - 'src/**/README.md'
      - 'dist-docs/**'
    acceptance_criteria:
      - id: AC-05-01-001
        text: 'npm run docs exits 0 and produces only generated-doc changes'
        validation: 'npm run docs; git status --short src/**/README.md dist-docs/'
      - id: AC-05-01-002
        text: 'No demo names or old racing symbols remain in generated READMEs'
        validation: 'git grep -Ein "(flappy|asciiMaze|neatchat|racing_curriculum|racing-curriculum|ant[ _-]?hive|predator/prey|RacingBatchOptions|evaluateRacingGeneration|network\.gpu\.racing)" src/**/README.md'
    parallelizable: false
    dependencies:
    next_slice: 05-02
  - slice_id: 05-02
    title: 'Write breaking-change migration note in RELEASE.md'
    status: '[PLANNED]'
    goal: documenting
    estimate_hours: 2
    files_to_change:
      - RELEASE.md
    acceptance_criteria:
      - id: AC-05-02-001
        text: 'RELEASE.md lists every removed file and symbol mapped to its replacement'
        validation: 'git grep -Ein "network\.gpu\.racing|RacingBatchOptions|evaluateRacingGeneration|RacingAgentRequest|evaluateConcurrentRacingAgents" RELEASE.md'
      - id: AC-05-02-002
        text: 'Migration note is discoverable from the top of RELEASE.md'
        validation: 'head -50 RELEASE.md'
    parallelizable: false
    dependencies:
      - 05-01
    next_slice: null
```

**User instruction:** Paste this full step packet.

**Step objective:** Ensure all generated documentation reflects the sanitized source and that external consumers have a clear migration note for the breaking rename.

**Context the agent must know:**

- `src/**/README.md` files are generated; do not hand-edit them.
- The migration note is the only remediation offered for the breaking API change.
- No aliases or shims will be added.

**Execution steps:**

1. Run `npm run docs`.
2. Verify generated READMEs with git grep.
3. Add a migration section to `RELEASE.md` mapping old ? new names.
4. Run lint and a final typecheck.

**Stop conditions:**

- Done when docs regenerate cleanly, generated READMEs are demo-free, and the migration note is present.
- Blocked if `npm run docs` fails or reintroduces demo names.

**Required validation:**

- `npm run docs`
- `git grep -Ein "(flappy|asciiMaze|neatchat|racing_curriculum|racing-curriculum|ant[ _-]?hive|predator/prey|RacingBatchOptions|evaluateRacingGeneration|network\.gpu\.racing)" src/**/README.md`
- `git grep -Ein "network\.gpu\.racing|RacingBatchOptions|evaluateRacingGeneration|RacingAgentRequest|evaluateConcurrentRacingAgents" RELEASE.md`

**Plan update requirement:** Record docs output, attach generated README diff summary, and advance to Step 06.

---
