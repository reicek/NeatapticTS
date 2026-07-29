# Orchestration System Fixes

**Status:** [WIP]

## Scope

This plan addresses twelve friction points discovered across fix-loop iterations (r5→r8) in the agentic workflow orchestration system. The work is confined to workflow skills, agent frontmatter, gate scripts, and plan-authoring conventions. It does **not** change core NEAT algorithms, public library APIs, or demo-specific logic.

The twelve issues are grouped into four phases:

- **Phase A — Policy & Gate Changes:** fix-severity classifier, single-expect relaxation, `04-implementing` targeted-test allowance, pre-specialist smoke gate, fix-loop convergence tracking.
- **Phase B — Infrastructure Fixes:** Cortex index auto-rebuild, slice context retention, test selection stored as file paths, plan-command-lint gate.
- **Phase C — Specialist Workflow Redesign:** shared validation phase, RAG-based fix-packet design, dispatch MCP prompt-length heuristic.
- **Phase D — Documentation & Concurrency:** concurrency clarification (real limit is 10, nested dispatch supported).

## Current state

Claim: 04-implementing @ 2026-07-29T01:10:00Z completing slice C2-green iteration 2 (coverage-gap closure); handoff to 05-green-testing.

- Phase A: [DONE] — all 6 steps complete and compressed to logs.
- Phase B: [DONE] — all 5 steps (01–05) green-validated and compressed to `plans/orchestration-fixes.logs.md`.
- Phase C: [WIP] — Phase C Step 02 Slice C2-green coverage gaps closed; pending 05-green-testing confirmation.
- Phase D: [PLANNED]

## Latest validation evidence

- green-light: true — Phase C plan readiness re-verified 2026-07-28 after Step 04 insertion; step-packet, plan-slice-quality, plan-command-lint, and validate-plan-sync gates all pass.
- Phase A: [DONE] — all steps compressed to `plans/orchestration-fixes.logs.md`.
- Phase B: [DONE] — all five steps green-validated and compressed to `plans/orchestration-fixes.logs.md`.
- Phase B Step 05 summary: `plan-command-lint.gate.mjs` 100% coverage, 46/46 tests pass, gate check on active plan returned 31 commands with 0 issues.
- Phase C Step 01: [DONE] — step-packet gate pass (0 violations), plan-slice-quality gate pass (0 violations).
- Phase C: [WIP] — next active frontier is Phase C Step 02 Slice C2-green (green validation).
- Phase C plan-readiness: green-light: true (verified 2026-07-28; step-packet pass, plan-slice-quality pass, slices ≤ 4h, ≤ 5 slices/step, IDs unique, DAG valid).
- Phase C Step 02 Slice C2-red-tests: [DONE] — red tests authored and failing for the expected reason (shared-validation.gate.mjs not yet implemented).
- Phase C Step 02 Slice C2-impl: [DONE] — shared-validation gate implemented; `specialist-review-workflow` and `execute` skills updated; focused Jest tests pass.
- Preflight C2-impl: `npx tsc --noEmit -p tsconfig.json` → exit 0; `npm run lint` → 0 errors, 44 pre-existing warnings; `npx prettier --check` passed on changed files.
- Focused Jest C2-impl: `npx jest --config=jest.config.mjs --no-cache scripts/agent-customization/gates/shared-validation.gate.test.ts` → 10/10 passed.
- Shared-validation gate smoke: `node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=scripts/agent-customization/gates/shared-validation.gate.mjs` → pass: true.
- Plan sync: `node .github/hooks/workflow-update-sync.mjs --plan=plans/orchestration-fixes.plans.md --json` → pass: true; `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/orchestration-fixes.plans.md` → pass: true.
- Phase C Step 04 packet: [PLANNED] — step packet, issue mapping, and AC-PLAN-005 wording updated for the dispatch MCP prompt-length heuristic.
- Gate outputs after Step 04 insertion:
  - `node scripts/agent-customization/gates/step-packet.gate.mjs --json` → pass: true, violations: [], planReadinessWarnings: []
  - `node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json` → pass: true
  - `node scripts/agent-customization/gates/plan-command-lint.gate.mjs --json --plan=plans/orchestration-fixes.plans.md` → pass: true
  - `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/orchestration-fixes.plans.md` → pass: true
- Phase C plan-readiness: green-light: true (re-verified 2026-07-28; step-packet pass, plan-slice-quality pass, plan-command-lint pass, validate-plan-sync pass; Step 04 slices ≤ 4h, ≤ 5 slices/step, IDs unique, DAG valid).
- fix-loop: C2-impl iteration 1 status=failed — specialist review returned REQUEST_CHANGES from implementation-pattern-scout and code-quality-auditor.
- fix-loop: C2-impl iteration 1 status=passed — all six requested changes addressed; preflight green; focused Jest 18/18 pass.
- Preflight C2-impl fix packet iteration 1:
  - `npx tsc --noEmit -p tsconfig.json` → exit 0
  - `npm run lint` → 0 errors, 44 pre-existing unrelated warnings
  - `npx prettier --check` → pass on changed files
  - Focused Jest: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/gates/shared-validation.gate.test.ts` → 18/18 pass
  - Focused Jest: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/gates/pre-specialist-smoke.gate.test.ts` → 16/16 pass
  - Routing table: `npm run agents:routing-table` → pass (changed=true)
  - Shared-validation gate smoke: `node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=scripts/agent-customization/gates/shared-validation.gate.mjs` → pass: true
  - Phase C Step 02 Slice C2-green: [DONE] — focused Jest 37/37 pass; coverage on `scripts/agent-customization/gates/shared-validation.gate.mjs` is statements 100%, branches 100%, functions 100%, lines 100%. AC-C2-008 and AC-C2-009 satisfied by iteration 2.
  - Preflight C2-green iteration 2:
    - `npx tsc --noEmit -p tsconfig.json` → exit 0
    - `npm run lint` → 0 errors, 44 pre-existing unrelated warnings
    - `npx prettier --check scripts/agent-customization/gates/shared-validation.gate.test.ts scripts/agent-customization/gates/shared-validation.gate.mjs plans/orchestration-fixes.plans.md` → pass
    - Focused Jest: `npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=shared-validation.gate.test.ts` → 37/37 pass, shared-validation.gate.mjs 100/100/100/100
    - Shared-validation gate smoke: `node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=scripts/agent-customization/gates/shared-validation.gate.mjs` → pass: true
    - Coverage gate: `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/shared-validation.gate.mjs` → pass: true, 100/100/100/100
    - Plan sync: `node .github/hooks/workflow-update-sync.mjs --plan=plans/orchestration-fixes.plans.md --json` → pass: true
    - Plan validation: `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/orchestration-fixes.plans.md` → pass: true, 0 issues
    - Step packet gate: `node scripts/agent-customization/gates/step-packet.gate.mjs --json` → pass: true, 0 violations
    - Plan slice quality gate: `node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json` → pass: true
    - Plan command lint gate: `node scripts/agent-customization/gates/plan-command-lint.gate.mjs --json --plan=plans/orchestration-fixes.plans.md` → pass: true, 0 issues
  - fix-loop: C2-green iteration 1 status=failed — coverage on shared-validation.gate.mjs below 100%.
  - fix-loop: C2-green iteration 2 status=passed — coverage gaps closed; handoff to 05-green-testing for coverage-guard confirmation.

```yaml
fix_packet:
  slice_id: C2-green
  iteration: 1
  status: FAILED
  goal: close-coverage-gaps
  observations:
    - source: 05-green-testing
      type: coverage-gap
      detail: 'shared-validation.gate.mjs lines 64-65: main CLI non-JSON output branch (console.log PASS/FAIL + fixHint) uncovered. Add test that invokes main() without --json flag.'
    - source: 05-green-testing
      type: coverage-gap
      detail: 'shared-validation.gate.mjs line 140: mkdirSync(artifactDir) branch uncovered because test artifact dirs already exist. Add test with a non-existent artifact directory path.'
    - source: 05-green-testing
      type: coverage-gap
      detail: 'shared-validation.gate.mjs line 188: defaultTestRunner early-return for empty testFiles array uncovered. Add test that passes empty changed-files list.'
    - source: 05-green-testing
      type: coverage-gap
      detail: 'shared-validation.gate.mjs lines 213-214: defaultTestRunner parsed.error branch uncovered. Add test with a mocked spawnSync that returns error on jest invocation.'
    - source: 05-green-testing
      type: coverage-gap
      detail: 'shared-validation.gate.mjs lines 217-220: JSON.parse failure fallback branch uncovered. Add test with mocked spawnSync returning non-JSON stdout.'
    - source: 05-green-testing
      type: coverage-gap
      detail: 'shared-validation.gate.mjs lines 239-241: default build runner result.error/result.signal fallback uncovered. Add test with mocked spawnSync returning error on build command.'
    - source: 05-green-testing
      type: coverage-gap
      detail: 'shared-validation.gate.mjs lines 260-262: default lint runner result.error/result.signal fallback uncovered. Add test with mocked spawnSync returning error on lint command.'
    - source: 05-green-testing
      type: coverage-gap
      detail: 'shared-validation.gate.mjs lines 278,290: parseChangedFiles --changed-files <value> space-separated form and empty-argv default not exercised. Add tests for both forms.'
    - source: 05-green-testing
      type: coverage-gap
      detail: 'shared-validation.gate.mjs lines 303,309: parseArtifactPath --artifact-path= form and default-return path not exercised. Add tests for both.'
```

```yaml
fix_packet:
  slice_id: C2-impl
  iteration: 1
  status: REQUEST_CHANGES
  observations:
    - source: implementation-pattern-scout
      type: protocol-inconsistency
      detail: 'execute/SKILL.md Section 5 loop step 2a still mandates only pre-specialist-smoke.gate.mjs, while Section 5.7 mandates shared-validation.gate.mjs. Reconcile Section 5 to reference the shared-validation gate as the first step before specialist review.'
    - source: implementation-pattern-scout
      type: helper-duplication
      detail: 'deriveTestFiles/isTestFile helpers are duplicated verbatim between pre-specialist-smoke.gate.mjs and shared-validation.gate.mjs. Extract to a shared module and import from both gates.'
    - source: implementation-pattern-scout
      type: test-style
      detail: 'shared-validation.gate.test.ts has multiple expect() per it() in several tests (artifact test: 5 expects, default-runner test: 4 expects). Decompose into single-expect tests where practical.'
    - source: code-quality-auditor
      type: coverage-config
      detail: 'Add scripts/agent-customization/gates/shared-validation.gate.mjs to jest.config.mjs collectCoverageFrom in the agent-customization-scripts project so the code-coverage gate can track it.'
    - source: code-quality-auditor
      type: routing-table-stale
      detail: 'Regenerate .github/agent-skill-routing-table.md via npm run agents:routing-table after skill edits.'
    - source: implementation-pattern-scout
      type: plan-rollback
      detail: 'Plan PlanUpdate rollback field uses git checkout which is prohibited. Replace with a non-git rollback description.'
  requested_changes:
    - 'Reconcile execute/SKILL.md Section 5 loop step 2a to reference shared-validation.gate.mjs before specialist review.'
    - 'Extract deriveTestFiles/isTestFile helpers to a shared module and import from both pre-specialist-smoke.gate.mjs and shared-validation.gate.mjs.'
    - 'Decompose multi-expect tests in shared-validation.gate.test.ts into single-expect tests where practical.'
    - 'Add shared-validation.gate.mjs to jest.config.mjs collectCoverageFrom (agent-customization-scripts project).'
    - 'Run npm run agents:routing-table to regenerate the routing table.'
    - 'Replace the git checkout rollback command in the C2-impl PlanUpdate block with a non-git description.'
```

```yaml
PlanUpdate:
  slice_id: C2-impl
  iteration: 1-fix
  changed_files:
    - scripts/agent-customization/gates/gate-test-utils.mjs
    - scripts/agent-customization/gates/pre-specialist-smoke.gate.mjs
    - scripts/agent-customization/gates/shared-validation.gate.mjs
    - scripts/agent-customization/gates/shared-validation.gate.test.ts
    - .github/skills/execute/SKILL.md
    - .github/agent-skill-routing-table.md
    - jest.config.mjs
    - plans/orchestration-fixes.plans.md
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check scripts/agent-customization/gates/gate-test-utils.mjs scripts/agent-customization/gates/pre-specialist-smoke.gate.mjs scripts/agent-customization/gates/shared-validation.gate.mjs scripts/agent-customization/gates/shared-validation.gate.test.ts .github/skills/execute/SKILL.md jest.config.mjs plans/orchestration-fixes.plans.md'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/gates/shared-validation.gate.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/gates/pre-specialist-smoke.gate.test.ts'
    - 'npm run agents:routing-table'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/gates/shared-validation.gate.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/gates/pre-specialist-smoke.gate.test.ts'
    - 'node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=scripts/agent-customization/gates/shared-validation.gate.mjs'
    - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json'
  rollback:
    - 'Manually revert changed files using the edit tool. Do NOT use git commands.'
  next: 'Run 05-green-testing on slice C2-green; attach coverage-guard evidence for shared-validation.gate.mjs and gate-test-utils.mjs'
  parallelizable: false
```

```yaml
PlanUpdate:
  slice_id: C2-green
  iteration: 2
  changed_files:
    - scripts/agent-customization/gates/shared-validation.gate.test.ts
    - plans/orchestration-fixes.plans.md
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check scripts/agent-customization/gates/shared-validation.gate.test.ts scripts/agent-customization/gates/shared-validation.gate.mjs plans/orchestration-fixes.plans.md'
    - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=shared-validation.gate.test.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/gates/shared-validation.gate.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=scripts/agent-customization/gates/shared-validation.gate.test.ts'
    - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/shared-validation.gate.mjs'
  rollback:
    - 'Manually revert changed files using the edit tool. Do NOT use git commands.'
  next: 'Run 05-green-testing and attach coverage-guard evidence for shared-validation.gate.mjs'
  parallelizable: false
```

<!-- B3 fix-loop history (r1-r5) and all Phase A/B2 PlanUpdate blocks compressed to plans/orchestration-fixes.logs.md -->

## Source of truth

- This file: `plans/orchestration-fixes.plans.md`
- Constitution: `plans/constitution.md`
- Routing table: `.github/agent-skill-routing-table.md`
- Active workstream: `plans/Neon_Shooter_NGE_Demo.plans.md` (downstream beneficiary)

## Plan-wide acceptance criteria

- id: AC-PLAN-001
  text: 'Every phase has Step 01–N packets with valid YAML per the step-packet schema.'
  validation: 'plans/orchestration-fixes.plans.md'
- id: AC-PLAN-002
  text: 'No step contains more than 5 slices and no slice exceeds 4 hours.'
  validation: 'scripts/agent-customization/gates/plan-slice-quality.gate.mjs'
- id: AC-PLAN-003
  text: 'Validation entries in the plan reference file paths, not Jest CLI flags (Issue 5 fix applied to the plan itself).'
  validation: 'plans/orchestration-fixes.plans.md'
- id: AC-PLAN-004
  text: 'Plan is registered in plans/README.md and placed in plans/Roadmap.md under the Meta-Workflow lane.'
  validation: 'plans/README.md'
- id: AC-PLAN-005
  text: 'All issues have at least one slice, step, or explicit skipped-record mapping.'
  validation: 'plans/orchestration-fixes.plans.md'

## Non-goals

- Re-architecting the five-tier delegation graph.
- Changing the model-routing table or agent frontmatter models.
- Modifying core NEAT/network code.
- Removing the specialist review gate entirely; only tuning its application.

## Open assumptions

- The repo will continue using Jest as the primary test runner.
- Cortex RAG remains backed by the Turso/libSQL pipeline.
- The real Copilot CLI concurrent agent limit is 10 (per Issue 12 clarification).

## Implementation phases

### Phase A — Policy & Gate Changes [DONE]

**Phase objective:** Update workflow policy and agent skills so trivial fixes move faster, tests can encode related assertions pragmatically, `04-implementing` can self-check with targeted tests, and fix loops escalate before they oscillate.

Phase A is complete. Detailed YAML packets, slice records, PlanUpdate history, and validation evidence for all six steps moved to `plans/orchestration-fixes.logs.md`.

#### Step 01: Plan Phase A step packets [DONE]

[DONE] Step 01: Plan Phase A step packets. Full YAML packet, acceptance criteria, and validation evidence moved to `plans/orchestration-fixes.logs.md`.

#### Step 02: Specialist review severity classifier [DONE]

[DONE] Step 02: Specialist review severity classifier implemented and green validated. See `plans/orchestration-fixes.logs.md` for the full YAML packet, acceptance criteria, slice details, and validation evidence.

#### Step 03: Relax single-expect rule [DONE]

[DONE] Step 03: Relax single-expect rule implemented and green validated. See `plans/orchestration-fixes.logs.md` for the full YAML packet, acceptance criteria, slice details, PlanUpdate history, and validation evidence.

#### Step 04: Allow 04-implementing targeted test runs [DONE]

[DONE] Step 04: Allow 04-implementing targeted test runs. All acceptance criteria AC-A4-001 through AC-A4-010 pass. See `plans/orchestration-fixes.logs.md` for the full YAML packet, acceptance criteria, slice details, PlanUpdate history, and validation evidence.

#### Step 05: Add pre-specialist smoke gate [DONE]

[DONE] Step 05: Add pre-specialist smoke gate implemented and green validated. See `plans/orchestration-fixes.logs.md` for the full YAML packet, acceptance criteria, slice details, PlanUpdate history, and validation evidence.

#### Step 06: Add fix-loop convergence tracking [DONE]

[DONE] Step 06: Add fix-loop convergence tracking. 24/24 focused tests pass, 100% coverage on `scripts/agent-customization/gates/convergence-tracker.gate.mjs`. Full YAML packet, slice details, PlanUpdate history, and validation evidence moved to `plans/orchestration-fixes.logs.md`.

---

### Phase B — Infrastructure Fixes [DONE]

**Phase objective:** Harden the RAG/infrastructure layer so Cortex stays fresh, completed slices remain retrievable, and plan-stored validation uses stable file paths instead of brittle CLI flags.

[DONE] Phase B complete. All five steps (01–05) validated and compressed. Full phase-level YAML packet, step packets, acceptance criteria, slices, traceability, PlanUpdate history, and validation evidence moved to `plans/orchestration-fixes.logs.md`.

#### Step 01: Plan Phase B step packets [DONE]

[DONE] Step 01: Phase B step packets (Steps 02–05) authored and validated via `step-packet` and `plan-slice-quality` gates. Full YAML packet, acceptance criteria, and validation evidence moved to `plans/orchestration-fixes.logs.md`.

#### Step 02: Cortex index auto-rebuild/freshness gate [DONE]

[DONE] Step 02: Cortex index auto-rebuild/freshness gate implemented and green validated. See plans/orchestration-fixes.logs.md for the full YAML packet, acceptance criteria, slice details, validation evidence, and fix-loop history.

#### Step 03: Slice context retention [DONE]

[DONE] Step 03: Slice context retention implemented and green validated (B3-green-r5 via 00-helping convergence escalation). See plans/orchestration-fixes.logs.md for the full YAML packet, acceptance criteria, traceability, slice details, validation evidence, and fix-loop history.

#### Step 04: Test selection as file paths [DONE]

[DONE] Step 04: Test selection stored as repo-relative file paths; validator updated, plan entries converted, green validation passed. Full YAML packet, slices, traceability, and evidence moved to plans/orchestration-fixes.logs.md.

#### Step 05: Plan-command-lint gate [DONE]

[DONE] Step 05: Plan-command-lint gate implemented and green validated; 100% coverage on `scripts/agent-customization/gates/plan-command-lint.gate.mjs`. Full YAML packet, acceptance criteria, traceability, slices, and validation evidence moved to `plans/orchestration-fixes.logs.md`.

---

### Phase C — Specialist Workflow Redesign [WIP]

**Phase objective:** Reduce specialist-review duplication and close the RAG gap for fix-loop packets by introducing a shared validation phase and a RAG-fix-packet convention.

```yaml
phase: C
title: 'Specialist Workflow Redesign'
status: '[WIP]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/orchestration-fixes.plans.md'
copy_paste: true
next_phase: 'Phase D — Documentation & Concurrency'
skills:
  - 'execute'
  - 'subagent-delegation-patterns'
  - 'routing-optimization-policy'
  - 'green-validation-gates'
validation:
  - 'plans/orchestration-fixes.plans.md'
  - '.github/skills/specialist-review-workflow/SKILL.md'
  - '.github/skills/execute/SKILL.md'
acceptance_criteria:
  - id: AC-C-001
    text: 'Specialist review is split into shared validation (tests+build run once) and perspective-specific review.'
    validation: '.github/skills/specialist-review-workflow/SKILL.md'
  - id: AC-C-002
    text: 'Shared validation results are attached to all specialists in a single artifact.'
    validation: '.github/skills/execute/SKILL.md'
  - id: AC-C-003
    text: 'Fix-loop packets can be loaded via RAG instead of embedding inline observations.'
    validation: '.github/skills/execute/SKILL.md'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
placeholder_steps:
  - 'Step 01 — Plan Phase C step packets'
  - 'Step 02 — Shared validation phase'
  - 'Step 03 — RAG-based fix-packet design'
  - 'Step 04 — Dispatch MCP prompt-length heuristic'
```

#### Step 01: Plan Phase C step packets [DONE]

```yaml
phase: C
step: 1
title: 'Plan Phase C step packets'
status: '[DONE]'
goal: 'planning'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/orchestration-fixes.plans.md'
copy_paste: true
next_step: 'Step 02 — Shared validation phase'
skills:
  - 'planning-acceptance-criteria'
  - 'phase-handoff-workflow'
  - 'tracker-handoff'
  - 'execute'
validation:
  - 'plans/orchestration-fixes.plans.md'
  - 'scripts/agent-customization/gates/step-packet.gate.mjs'
  - 'scripts/agent-customization/gates/plan-slice-quality.gate.mjs'
acceptance_criteria:
  - id: AC-C1-001
    text: 'Step packets for Phase C Steps 02–03 are authored and pass the step-packet gate.'
    validation: 'scripts/agent-customization/gates/step-packet.gate.mjs'
  - id: AC-C1-002
    text: 'Slice-quality gate confirms no slice exceeds 4 hours and no step exceeds 5 slices.'
    validation: 'scripts/agent-customization/gates/plan-slice-quality.gate.mjs'
constitution_check:
  - 'principle-5-unique-ids'
```

[DONE] Step 01: Phase C Step 02–03 packets authored; step-packet and plan-slice-quality gates pass with 0 violations.

#### Step 02: Shared validation phase [WIP]

```yaml
phase: C
step: 2
title: 'Shared validation phase'
status: '[WIP]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/orchestration-fixes.plans.md'
copy_paste: true
next_step: 'Step 03 — RAG-based fix-packet design'
skills:
  - 'execute'
  - 'subagent-delegation-patterns'
  - 'routing-optimization-policy'
  - 'green-validation-gates'
validation:
  - '.github/skills/specialist-review-workflow/SKILL.md'
  - '.github/skills/execute/SKILL.md'
  - 'scripts/agent-customization/gates/shared-validation.gate.mjs'
  - 'scripts/agent-customization/gates/shared-validation.gate.test.ts'
acceptance_criteria:
  - id: AC-C2-001
    text: 'Specialist-review-workflow skill defines a shared validation phase before perspective-specific review.'
    validation: '.github/skills/specialist-review-workflow/SKILL.md'
  - id: AC-C2-002
    text: 'Shared-validation gate runs tests+build once and emits a structured artifact.'
    validation: 'scripts/agent-customization/gates/shared-validation.gate.mjs'
  - id: AC-C2-003
    text: 'Execute skill documents that specialists receive the shared artifact instead of re-running tests.'
    validation: '.github/skills/execute/SKILL.md'
constitution_check:
  - 'principle-5-unique-ids'
slices:
  - slice_id: 'C2-red-tests'
    title: 'Red tests for shared validation gate'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'scripts/agent-customization/gates/shared-validation.gate.test.ts'
    acceptance_criteria:
      - id: AC-C2-004
        text: 'Red tests exist and fail before the gate is implemented.'
        validation: 'scripts/agent-customization/gates/shared-validation.gate.test.ts'
    parallelizable: false
    dependencies: []
    next_slice: 'C2-impl'
    validation_evidence:
      - command: 'npx jest --config=jest.config.mjs --no-cache scripts/agent-customization/gates/shared-validation.gate.test.ts'
        result: 'FAIL — 10/10 tests failed because shared-validation.gate.mjs does not yet exist'
        note: 'Red contract confirmed: module-not-found errors are the expected failure mode before C2-impl'
  - slice_id: 'C2-impl'
    title: 'Implement shared validation gate and skill updates'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'scripts/agent-customization/gates/shared-validation.gate.mjs'
      - '.github/skills/specialist-review-workflow/SKILL.md'
      - '.github/skills/execute/SKILL.md'
    acceptance_criteria:
      - id: AC-C2-005
        text: 'Gate accepts a list of changed file paths, runs focused tests and build once, and writes a JSON artifact.'
        validation: 'scripts/agent-customization/gates/shared-validation.gate.mjs'
      - id: AC-C2-006
        text: 'Specialist-review-workflow skill documents shared validation + perspective review phases.'
        validation: '.github/skills/specialist-review-workflow/SKILL.md'
      - id: AC-C2-007
        text: 'Execute skill updates the pre-green specialist review protocol to use the shared artifact.'
        validation: '.github/skills/execute/SKILL.md'
    parallelizable: false
    dependencies:
      - 'C2-red-tests'
    next_slice: 'C2-green'
  - slice_id: 'C2-green'
    title: 'Green validation for shared validation phase'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'scripts/agent-customization/gates/shared-validation.gate.test.ts'
      - 'coverage/lcov.info'
    acceptance_criteria:
      - id: AC-C2-008
        text: 'All red tests pass.'
        validation: 'scripts/agent-customization/gates/shared-validation.gate.test.ts'
      - id: AC-C2-009
        text: '100% coverage on the shared-validation gate script.'
        validation: 'scripts/agent-customization/gates/shared-validation.gate.mjs'
    parallelizable: false
    dependencies:
      - 'C2-impl'
    next_slice: null
    validation_evidence:
      - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=shared-validation.gate.test.ts'
        result: 'PASS — 37/37 tests passed'
        note: 'AC-C2-008 satisfied after coverage-gap closure'
      - command: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=shared-validation.gate.test.ts'
        result: 'PASS — shared-validation.gate.mjs coverage: statements 100%, branches 100%, functions 100%, lines 100%'
        note: 'AC-C2-009 satisfied'
      - command: 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/shared-validation.gate.mjs'
        result: 'PASS — shared-validation.gate.mjs lines/statements/functions/branches = 100/100/100/100'
        note: 'Coverage gate confirms AC-C2-009'
```

**User instruction:** Paste this full step packet.

**Step objective:** Fix Issue 11 by splitting specialist review into (a) a shared validation phase that runs tests and build once, and (b) perspective-specific review where each specialist only reads code + shared results.

**Context the agent must know:**

- Issue 11: 3 specialists independently run the same Jest suites and build.
- The shared artifact must be JSON with test results, build result, changed files, and lint status.
- Specialists should not re-run the suite; they review the artifact plus the code.

**Execution steps:**

1. Red tests for the shared-validation gate.
2. Implement the gate and update `specialist-review-workflow` and `execute` skills.
3. Green validation.

**Stop conditions:**

- Done: gate passes with 100% coverage and skills are updated.
- Blocked: cannot produce a deterministic shared artifact path.
- Route-back: if red tests are wrong, return to `03-red-testing`.

**Required validation:**

- `npx jest --config=jest.config.mjs --no-cache scripts/agent-customization/gates/shared-validation.gate.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --coverage scripts/agent-customization/gates/shared-validation.gate.mjs`

**Plan update requirement:** Record gate and skill evidence.

#### Step 03: RAG-based fix-packet design [PLANNED]

```yaml
phase: C
step: 3
title: 'RAG-based fix-packet design'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/orchestration-fixes.plans.md'
copy_paste: true
next_step: 'Step 04 — Dispatch MCP prompt-length heuristic'
skills:
  - 'execute'
  - 'tracker-handoff'
  - 'plan-sync-validation'
validation:
  - '.github/skills/execute/SKILL.md'
  - '.github/skills/tracker-handoff/SKILL.md'
  - 'plans/orchestration-fixes.plans.md'
acceptance_criteria:
  - id: AC-C3-001
    text: 'Execute skill documents a RAG-based fix-packet convention that stores observations in the plan instead of inline prompts.'
    validation: '.github/skills/execute/SKILL.md'
  - id: AC-C3-002
    text: 'Tracker-handoff skill defines a fix-packet section shape under VALIDATION_EVIDENCE.'
    validation: '.github/skills/tracker-handoff/SKILL.md'
  - id: AC-C3-003
    text: 'This plan contains at least one example fix-packet section using the new shape.'
    validation: 'plans/orchestration-fixes.plans.md'
constitution_check:
  - 'principle-5-unique-ids'
slices:
  - slice_id: 'C3-design-rag-fix-packet'
    title: 'Design RAG fix-packet section shape'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - '.github/skills/execute/SKILL.md'
      - '.github/skills/tracker-handoff/SKILL.md'
    acceptance_criteria:
      - id: AC-C3-004
        text: 'Execute skill specifies that fix-loop observations are appended to the plan under a deterministic fix-packet ID.'
        validation: '.github/skills/execute/SKILL.md'
      - id: AC-C3-005
        text: 'Tracker-handoff skill documents the fix-packet YAML schema and evidence section.'
        validation: '.github/skills/tracker-handoff/SKILL.md'
    parallelizable: false
    dependencies: []
    next_slice: 'C3-example-in-plan'
  - slice_id: 'C3-example-in-plan'
    title: 'Add example fix-packet section to this plan'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'plans/orchestration-fixes.plans.md'
    acceptance_criteria:
      - id: AC-C3-006
        text: 'This plan includes a sample fix-packet block with slice_id, observations, and requested_changes fields.'
        validation: 'plans/orchestration-fixes.plans.md'
    parallelizable: false
    dependencies:
      - 'C3-design-rag-fix-packet'
    next_slice: 'C3-green'
  - slice_id: 'C3-green'
    title: 'Green validation for RAG fix-packet design'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - '.github/skills/execute/SKILL.md'
      - '.github/skills/tracker-handoff/SKILL.md'
      - 'plans/orchestration-fixes.plans.md'
    acceptance_criteria:
      - id: AC-C3-007
        text: 'Markdown lint passes on changed skill files.'
        validation: '.github/skills/execute/SKILL.md'
      - id: AC-C3-008
        text: 'Plan passes step-packet gate after adding the example fix-packet section.'
        validation: 'plans/orchestration-fixes.plans.md'
    parallelizable: false
    dependencies:
      - 'C3-example-in-plan'
    next_slice: null
```

**User instruction:** Paste this full step packet.

**Step objective:** Fix Issue 10 by designing a RAG-based fix-packet convention so fix-loop observations live in the plan/slice context instead of inline prompts.

**Context the agent must know:**

- Issue 10: the RAG principle forbids inline instructions, but fix packets necessarily embed compiled observations.
- The fix: store observations under a `fix_packet` section in the plan/slice, keyed by slice_id and iteration.
- The orchestrator dispatches `04-implementing` with only the slice_id; the agent loads the fix packet via `get_slice_context`.

**Execution steps:**

1. Design the fix-packet schema in `execute` and `tracker-handoff` skills.
2. Add an example fix-packet section to this plan.
3. Validate skills and plan.

**Stop conditions:**

- Done: skills document the convention and this plan includes a valid example.
- Blocked: schema conflicts with existing `VALIDATION_EVIDENCE` shape.
- Route-back: if the example breaks the step-packet gate, revise.

**Required validation:**

- `npm run lint`
- `node scripts/agent-customization/gates/step-packet.gate.mjs --json`

**Plan update requirement:** Record the new convention and example in the plan.

#### Step 04: Dispatch MCP prompt-length heuristic [PLANNED]

```yaml
phase: C
step: 4
title: 'Dispatch MCP prompt-length heuristic'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/orchestration-fixes.plans.md'
copy_paste: true
next_step: 'Phase D Step 01 — Plan documentation & concurrency packets'
skills:
  - 'execute'
  - 'red-test-contracts'
  - 'green-validation-gates'
validation:
  - 'scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs'
  - 'scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs'
  - '.github/skills/execute/SKILL.md'
acceptance_criteria:
  - id: AC-C4-001
    text: 'build_dispatch_packet rejects any prompt whose string length exceeds the configured maximum (200 characters).'
    validation: 'scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs'
  - id: AC-C4-002
    text: 'The rejection response includes measured prompt_length and prompt_length_max fields.'
    validation: 'scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs'
  - id: AC-C4-003
    text: 'get_dispatch_policy exposes the prompt-length rule and the current character limit.'
    validation: 'scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs'
  - id: AC-C4-004
    text: 'All existing short-prompt example prompts continue to return dispatch_allowed: true with the same packet shape.'
    validation: 'scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs'
  - id: AC-C4-005
    text: 'The length limit is defined by a single named constant in the dispatch MCP source, not magic numbers.'
    validation: 'scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs'
  - id: AC-C4-006
    text: 'execute/SKILL.md documents the prompt-length rejection in the build_dispatch_packet common-failure reasons.'
    validation: '.github/skills/execute/SKILL.md'
  - id: AC-C4-007
    text: 'execute/SKILL.md RAG-Based Dispatch Policy notes that neataptic-dispatch-mcp mechanically enforces the short-prompt rule.'
    validation: '.github/skills/execute/SKILL.md'
  - id: AC-C4-008
    text: 'The dispatch MCP --self-check passes after the guard is added.'
    validation: 'node scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs --self-check --json'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: 'C4-red-tests'
    title: 'Red tests for prompt-length guard'
    status: '[PLANNED]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs'
    acceptance_criteria:
      - id: AC-C4-101
        text: 'A new test calls build_dispatch_packet with a 201-character prompt and asserts ok=false, dispatch_allowed=false, and a reason mentioning prompt length.'
        validation: 'scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs'
      - id: AC-C4-102
        text: 'A new test calls get_dispatch_policy and asserts the response exposes prompt_length_rule and prompt_length_max.'
        validation: 'scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs'
      - id: AC-C4-103
        text: 'Existing short-prompt tests are adjusted, if necessary, to tolerate the new policy fields without failing before implementation.'
        validation: 'scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs'
    parallelizable: false
    dependencies: []
    next_slice: 'C4-impl'
  - slice_id: 'C4-impl'
    title: 'Implement prompt-length guard and skill backstop'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs'
      - '.github/skills/execute/SKILL.md'
    acceptance_criteria:
      - id: AC-C4-201
        text: 'build_dispatch_packet rejects prompts longer than 200 characters with a clear reason and prompt_length fields.'
        validation: 'scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs'
      - id: AC-C4-202
        text: 'get_dispatch_policy and the server self-check include the prompt-length rule and limit.'
        validation: 'scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs'
      - id: AC-C4-203
        text: 'execute/SKILL.md documents the new failure reason and references the MCP guard in the RAG-Based Dispatch Policy.'
        validation: '.github/skills/execute/SKILL.md'
    parallelizable: false
    dependencies:
      - 'C4-red-tests'
    next_slice: 'C4-green'
  - slice_id: 'C4-green'
    title: 'Green validation for prompt-length guard'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs'
      - 'coverage/lcov.info'
    acceptance_criteria:
      - id: AC-C4-301
        text: 'All red tests pass (overlong rejection, policy exposure, short-prompt preservation).'
        validation: 'node --test scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs'
      - id: AC-C4-302
        text: 'Dispatch MCP --self-check exits cleanly.'
        validation: 'node scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs --self-check --json'
      - id: AC-C4-303
        text: 'Plan passes step-packet and plan-slice-quality gates after Step 04 is inserted.'
        validation: 'node scripts/agent-customization/gates/step-packet.gate.mjs --json'
    parallelizable: false
    dependencies:
      - 'C4-impl'
    next_slice: null
```

**User instruction:** Paste this full step packet.

**Step objective:** Add a prompt-length guard to `neataptic-dispatch-mcp / build_dispatch_packet` so the dispatch MCP mechanically enforces the short-prompt RAG contract in `execute/SKILL.md`.

**Context the agent must know:**

- The RAG-based dispatch policy requires prompts to contain only a slice/step ID and a one-line RAG load instruction.
- The dispatch MCP currently validates delegation legality and echoes the prompt verbatim, but does not limit prompt length.
- The guard must reject prompts longer than 200 characters and include `prompt_length` / `prompt_length_max` diagnostics in the response.
- Existing short example prompts in `execute/SKILL.md` are all under the limit and must remain valid.
- `get_dispatch_policy` should expose the rule so orchestrators can discover the limit without probing.

**Execution steps:**

1. Author red tests that fail because the guard does not yet exist.
2. Implement the guard, expose the rule in `get_dispatch_policy`, and document it in `execute/SKILL.md`.
3. Run green validation (Node native tests, self-check, and plan gates).

**Stop conditions:**

- Done: overlong prompts are rejected, short prompts still dispatch, policy is discoverable, and skills are documented.
- Blocked: the guard breaks existing short-prompt examples or the self-check.
- Route-back: if the test file needs a different harness, return to `03-red-testing`.

**Required validation:**

- `node --test scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs`
- `node scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs --self-check --json`
- `node scripts/agent-customization/gates/step-packet.gate.mjs --json`
- `node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json`

**Plan update requirement:** Record guard behavior, threshold, and green evidence.

---

### Phase D — Documentation & Concurrency [PLANNED]

**Phase objective:** Document that the real concurrent agent limit is 10 and that nested dispatch is supported; clarify RAG principle exceptions for fix packets.

```yaml
phase: D
title: 'Documentation & Concurrency'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/orchestration-fixes.plans.md'
copy_paste: true
next_phase: 'Archive plan and create matching .logs.md'
skills:
  - 'educational-docs'
  - 'execute'
  - 'subagent-delegation-patterns'
  - 'plan-sync-validation'
validation:
  - 'plans/orchestration-fixes.plans.md'
  - '.github/skills/execute/SKILL.md'
  - '.github/skills/subagent-delegation-patterns/SKILL.md'
acceptance_criteria:
  - id: AC-D-001
    text: 'Execute skill documents concurrency limit of 10 and nested dispatch support.'
    validation: '.github/skills/execute/SKILL.md'
  - id: AC-D-002
    text: 'Subagent-delegation-patterns skill removes any wording that assumes a limit of 2 for nested dispatch.'
    validation: '.github/skills/subagent-delegation-patterns/SKILL.md'
  - id: AC-D-003
    text: 'RAG principle exception for fix packets is documented as a controlled deviation with required plan storage.'
    validation: '.github/skills/execute/SKILL.md'
constitution_check:
  - 'principle-4-small-slices'
placeholder_steps:
  - 'Step 01 — Plan Phase D step packets'
  - 'Step 02 — Concurrency clarification and RAG exception docs'
```

#### Step 01: Plan Phase D step packets [PLANNED]

```yaml
phase: D
step: 1
title: 'Plan Phase D step packets'
status: '[PLANNED]'
goal: 'planning'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/orchestration-fixes.plans.md'
copy_paste: true
next_step: 'Step 02 — Concurrency clarification and RAG exception docs'
skills:
  - 'planning-acceptance-criteria'
  - 'phase-handoff-workflow'
  - 'tracker-handoff'
  - 'execute'
validation:
  - 'plans/orchestration-fixes.plans.md'
  - 'scripts/agent-customization/gates/step-packet.gate.mjs'
  - 'scripts/agent-customization/gates/plan-slice-quality.gate.mjs'
acceptance_criteria:
  - id: AC-D1-001
    text: 'Step packet for Phase D Step 02 is authored and passes the step-packet gate.'
    validation: 'scripts/agent-customization/gates/step-packet.gate.mjs'
  - id: AC-D1-002
    text: 'Slice-quality gate confirms Step 02 has at most 5 slices.'
    validation: 'scripts/agent-customization/gates/plan-slice-quality.gate.mjs'
constitution_check:
  - 'principle-5-unique-ids'
```

**User instruction:** Paste this full step packet.

**Step objective:** Author the remaining Phase D step packet (Step 02) before implementation begins.

**Context the agent must know:**

- Phase D is documentation-only for Issue 12 and RAG-clarification for Issue 10.
- Issue 12: real concurrency limit is 10, nested dispatch is supported.

**Execution steps:**

1. Author Step 02 YAML packet.
2. Run `step-packet` and `plan-slice-quality` gates.

**Stop conditions:**

- Done: packet authored and gates pass.
- Blocked: schema violations.

**Required validation:**

- `node scripts/agent-customization/gates/step-packet.gate.mjs --json`
- `node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json`

**Plan update requirement:** Update Phase D with the new step packet and validation evidence.

#### Step 02: Concurrency clarification and RAG exception docs [PLANNED]

```yaml
phase: D
step: 2
title: 'Concurrency clarification and RAG exception docs'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/orchestration-fixes.plans.md'
copy_paste: true
next_step: 'Phase compression and archive'
skills:
  - 'educational-docs'
  - 'execute'
  - 'subagent-delegation-patterns'
validation:
  - '.github/skills/execute/SKILL.md'
  - '.github/skills/subagent-delegation-patterns/SKILL.md'
acceptance_criteria:
  - id: AC-D2-001
    text: 'Execute skill Section 5.5 documents concurrency=10 and nested dispatch support.'
    validation: '.github/skills/execute/SKILL.md'
  - id: AC-D2-002
    text: 'Subagent-delegation-patterns skill does not state nested dispatch is impossible.'
    validation: '.github/skills/subagent-delegation-patterns/SKILL.md'
  - id: AC-D2-003
    text: 'Execute skill documents the RAG fix-packet exception and its required plan-storage condition.'
    validation: '.github/skills/execute/SKILL.md'
constitution_check:
  - 'principle-4-small-slices'
slices:
  - slice_id: 'D2-concurrency-docs'
    title: 'Document concurrency=10 and nested dispatch support'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - '.github/skills/execute/SKILL.md'
      - '.github/skills/subagent-delegation-patterns/SKILL.md'
    acceptance_criteria:
      - id: AC-D2-004
        text: 'Execute skill explicitly states the real limit is 10 and nested dispatch is supported.'
        validation: '.github/skills/execute/SKILL.md'
      - id: AC-D2-005
        text: 'Subagent-delegation-patterns skill removes or corrects any 2-slot nested-dispatch impossibility claim.'
        validation: '.github/skills/subagent-delegation-patterns/SKILL.md'
    parallelizable: true
    dependencies: []
    next_slice: 'D2-rag-exception-docs'
  - slice_id: 'D2-rag-exception-docs'
    title: 'Document RAG exception for fix packets'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - '.github/skills/execute/SKILL.md'
    acceptance_criteria:
      - id: AC-D2-006
        text: 'Execute skill has a section stating fix-loop observations may be stored in the plan and loaded via RAG.'
        validation: '.github/skills/execute/SKILL.md'
      - id: AC-D2-007
        text: 'The exception requires a deterministic slice_id and a plan-stored fix_packet block.'
        validation: '.github/skills/execute/SKILL.md'
    parallelizable: true
    dependencies: []
    next_slice: 'D2-green'
  - slice_id: 'D2-green'
    title: 'Green validation for documentation updates'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - '.github/skills/execute/SKILL.md'
      - '.github/skills/subagent-delegation-patterns/SKILL.md'
    acceptance_criteria:
      - id: AC-D2-008
        text: 'Markdown lint passes on changed skill files.'
        validation: '.github/skills/execute/SKILL.md'
      - id: AC-D2-009
        text: 'No contradictions remain between execute and subagent-delegation-patterns skills.'
        validation: '.github/skills/subagent-delegation-patterns/SKILL.md'
    parallelizable: false
    dependencies:
      - 'D2-concurrency-docs'
      - 'D2-rag-exception-docs'
    next_slice: null
```

**User instruction:** Paste this full step packet.

**Step objective:** Fix Issue 12 by documenting that the real concurrency limit is 10 and nested dispatch is supported; also document the RAG exception for fix packets.

**Context the agent must know:**

- Issue 12: the current policy documents concurrency-limits but was written when the limit appeared to be 2. The real limit is 10.
- No code change is needed; only skill documentation.
- The RAG exception for fix packets is tied to Phase C Step 03.

**Execution steps:**

1. Update `execute` skill concurrency section to state limit=10 and nested dispatch is supported.
2. Update `subagent-delegation-patterns` skill to remove any 2-slot impossibility claim.
3. Document the RAG fix-packet exception in `execute` skill.
4. Run lint and search for contradictions.

**Stop conditions:**

- Done: skills are consistent and lint passes.
- Blocked: other skills contradict the concurrency/RAG documentation.
- Route-back: if contradictions remain, escalate to `00-helping`.

**Required validation:**

- `npm run lint`
- Cortex search for "concurrent limit" / "nested dispatch" contradictions.

**Plan update requirement:** Record documentation diffs and lint result.

---

## Validation gates

All phase work must pass the following gates before being marked [DONE]:

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/orchestration-fixes.plans.md`
- `node scripts/agent-customization/gates/step-packet.gate.mjs --json`
- `node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json`
- `node scripts/agent-customization/gates/plan-command-lint.gate.mjs --json --plan=plans/orchestration-fixes.plans.md`

## Issue-to-slice mapping

| Issue | Suggested remediation                                       | Phase | Step | Slice(s)                                                                     |
| ----- | ----------------------------------------------------------- | ----- | ---- | ---------------------------------------------------------------------------- |
| 1     | Specialist review severity classifier (1 vs 3+ specialists) | A     | 02   | A2-red-tests, A2-impl, A2-green                                              |
| 4     | Relax single-expect rule                                    | A     | 03   | A3-relax-red-contracts, A3-relax-creating-unit-tests, A3-green               |
| 6     | Allow `04-implementing` targeted test runs                  | A     | 04   | A4-update-execute-skill, A4-update-impl-standards, A4-update-agent, A4-green |
| 7     | Pre-specialist smoke gate                                   | A     | 05   | A5-red-tests, A5-impl, A5-green                                              |
| 8     | Fix-loop convergence tracking                               | A     | 06   | A6-red-tests, A6-impl, A6-green                                              |
| 3     | Cortex index auto-rebuild/freshness gate                    | B     | 02   | B2-red-tests, B2-impl, B2-green                                              |
| 2     | Slice context retention                                     | B     | 03   | B3-red-tests, B3-impl, B3-green                                              |
| 5     | Test selection as file paths                                | B     | 04   | B4-impl-validator, B4-apply-to-plan, B4-green                                |
| 9     | Plan-command-lint gate                                      | B     | 05   | B5-red-tests, B5-impl, B5-green                                              |
| 11    | Shared validation phase for specialists                     | C     | 02   | C2-red-tests, C2-impl, C2-green                                              |
| 10    | RAG-based fix-packet design                                 | C     | 03   | C3-design-rag-fix-packet, C3-example-in-plan, C3-green                       |
| 13    | Prompt-length guard for dispatch MCP                        | C     | 04   | C4-red-tests, C4-impl, C4-green                                              |
| 12    | Concurrency=10 / nested dispatch docs                       | D     | 02   | D2-concurrency-docs, D2-rag-exception-docs, D2-green                         |

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history. Load context via Cortex MCP and any declared pre_execute_hook/get_slice_context.

Context: Orchestration System Fixes — Phase A [DONE]; Phase B — Infrastructure Fixes [DONE]; Phase C — Specialist Workflow Redesign [WIP].
Changed files:
- plans/orchestration-fixes.plans.md
- plans/orchestration-fixes.logs.md
What is already covered: Phase A [DONE] and compressed. Phase B Steps 01–05 [DONE] and compressed; all green validations passed, including B5-green 100% coverage on `plan-command-lint.gate.mjs`. Phase B phase-level YAML packet moved to plans/orchestration-fixes.logs.md.
Next narrow task: Execute Phase C Step 02 — Shared validation phase. Implement the shared-validation gate and update `specialist-review-workflow` and `execute` skills, then run red/green validation.
Required validations:
- node scripts/agent-customization/gates/step-packet.gate.mjs --json
- node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json
Known worktree cautions:
- This is a meta-workflow plan; code changes are in .github/skills/, .github/agents/, scripts/agent-customization/gates/, and scripts/agent-customization/mcp/.
- Do not change core NEAT/network code or demo-specific logic.
- Pre-existing gate exceptions (agent-frontmatter unknown skills, cortex-index workflow_mcp_alive after rebuild) are recorded in .github/ai-learning/learning-log.jsonl and are outside Phase C scope unless touched.
```
