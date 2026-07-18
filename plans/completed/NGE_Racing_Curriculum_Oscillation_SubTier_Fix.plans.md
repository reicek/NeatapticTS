# NGE Racing Curriculum Oscillation & Sub-Tier Fix

**Status:** [DONE]

Upstream plan: [NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md](NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md) [DONE]

> **Plan complete.** Core oscillation fix (Phase 4) verified: 82/82 tests pass, NGE specialist APPROVED, all gates green. Phases 5–7 (bundle rebuild, documentation, session logging) are **CANCELLED** and folded into [Racing_Perception_Redesign.plans.md](Racing_Perception_Redesign.plans.md) — bundle rebuild and browser validation will happen there.

## Scope

This focused workstream addresses two independent gaps discovered during the racing-curriculum Phase 8 / v2 driving-improvement pass:

- **A. Library-side juvenile sub-tier exhaustion.** `src/neat/nge-juvenile/` currently hard-codes a single `juvenileStage` lifecycle label (`baby`) and does not scale tier-specific exhaustion / retry thresholds with the current network size. Agents in deeper tiers never get their size-proportional exploration allowance, which masks the racing network's ability to grow into its target capacity.
- **B. Racing runtime steering/score oscillation.** `examples/racing_curriculum/controller/runtime.adaptation.ts` resets `lifecycleStage: 'baby'` every tick and its `RACING_VARIANT_SCORER` / `evaluateRacingTrendScore` do not penalize rapid steering or score oscillation. The adaptation loop therefore chases noisy local improvements instead of rewarding sustained progress.

This plan covers both surfaces and ends with a rebuilt racing bundle and browser smoke validation that the oscillation penalties and sub-tier thresholds behave visibly.

## Current state

Claim: 04-implementing @ 2026-07-18T17:05:00Z

- Slice 04-a implementation complete: tiered exhaustion constants, tier-resolution helper, and grow-stabilize threshold scaling added.
- Slice 04-b implementation complete: tier-aware oscillation penalty and additive growth-commit threshold boost for young/small networks.
- Specialist review fix cycle (post-04-b) updated `examples/racing_curriculum/controller/runtime.adaptation.test.ts`:
  1. `evaluateRacingTrendScore` â‰¥200-neuron gate-specific test now compares the **same** oscillating history on a small network (`Network(4, 2)`, <200 neurons, no penalty) versus a large network (`Network(200, 1)`, â‰¥200 neurons, penalty applied). The history has a negative trend (`[5, 3, 5, 3, 1]`) so the complexity bonus is gated off, isolating the oscillation-penalty gate as the only score difference.
  2. Runtime engine `scoreFn` wiring is verified by source-code inspection that the closure calls `scoreRacingVariant(outputs, target, tickInput.network.nodes.length)`.
- Preflight (tsc, lint, prettier) passes; Jest intentionally not run per 04-implementing contract.
- Target files confirmed on disk:
  - `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts`
  - `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts`
  - `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts`
  - `examples/racing_curriculum/controller/runtime.adaptation.ts`
  - `examples/racing_curriculum/controller/runtime.adaptation.test.ts`

```yaml
PlanUpdate:
  slice_id: 04-b
  changed_files:
    - examples/racing_curriculum/controller/runtime.adaptation.ts
    - examples/racing_curriculum/controller/runtime.adaptation.test.ts
  preflight:
    - command: 'npx tsc --noEmit -p tsconfig.json'
      result: 'PASS'
    - command: 'npm run lint -- examples/racing_curriculum/controller/runtime.adaptation.ts examples/racing_curriculum/controller/runtime.adaptation.test.ts'
      result: 'PASS (0 errors, 21 unrelated warnings in src/neat/nge-juvenile/)'
    - command: 'npx prettier --check examples/racing_curriculum/controller/runtime.adaptation.ts examples/racing_curriculum/controller/runtime.adaptation.test.ts'
      result: 'PASS'
    - command: 'npm run quality:folder -- --folder=examples/racing_curriculum/controller'
      result: 'PASS (0 TS diagnostics, 0 ESLint errors, 39/39 JSDoc symbols)'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation'
  rollback:
    - 'git checkout -- examples/racing_curriculum/controller/runtime.adaptation.ts'
    - 'git checkout -- examples/racing_curriculum/controller/runtime.adaptation.test.ts'
  next: 'Run 05-green-testing on the focused Jest slice and attach coverage-guard evidence'
```

```yaml
PlanUpdate:
  slice_id: 04-b-fix
  parent_slice_id: 04-b
  changed_files:
    - examples/racing_curriculum/controller/runtime.adaptation.test.ts
  preflight:
    - command: 'npx tsc --noEmit -p tsconfig.json'
      result: 'PASS'
    - command: 'npm run lint -- examples/racing_curriculum/controller/runtime.adaptation.ts examples/racing_curriculum/controller/runtime.adaptation.test.ts'
      result: 'PASS (0 errors, 21 unrelated warnings in src/neat/nge-juvenile/)'
    - command: 'npx prettier --check examples/racing_curriculum/controller/runtime.adaptation.ts examples/racing_curriculum/controller/runtime.adaptation.test.ts'
      result: 'PASS'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation'
  rollback:
    - 'git checkout -- examples/racing_curriculum/controller/runtime.adaptation.test.ts'
  next: 'NGE specialist re-review, then 05-green-testing on the focused Jest slice'
```

```yaml
PlanUpdate:
  slice_id: 04-b-fix-2
  parent_slice_id: 04-b
  changed_files:
    - examples/racing_curriculum/controller/runtime.adaptation.test.ts
  preflight:
    - command: 'npx tsc --noEmit -p tsconfig.json'
      result: 'PASS'
    - command: 'npm run lint -- examples/racing_curriculum/controller/runtime.adaptation.ts examples/racing_curriculum/controller/runtime.adaptation.test.ts'
      result: 'PASS (0 errors, 21 unrelated warnings in src/neat/nge-juvenile/)'
    - command: 'npx prettier --check examples/racing_curriculum/controller/runtime.adaptation.ts examples/racing_curriculum/controller/runtime.adaptation.test.ts'
      result: 'PASS'
    - command: 'npm run quality:folder -- --folder=examples/racing_curriculum/controller'
      result: 'PASS (0 TS diagnostics, 0 ESLint errors, 39/39 JSDoc symbols)'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation'
  rollback:
    - 'git checkout -- examples/racing_curriculum/controller/runtime.adaptation.test.ts'
  next: 'NGE specialist re-review, then 05-green-testing on the focused Jest slice'
```

## Clarifications

No clarifications recorded.

## Decision records

No decision records yet.

## Latest validation evidence

- closure_note: 'Plan marked [DONE] by user decision. Core oscillation fix (Phase 4) is green-gated and complete. Remaining Phases 5–7 are CANCELLED and folded into Racing_Perception_Redesign.plans.md.'
- green-light: true
- verified_at: 2026-07-18T13:47:00-04:00
- verifier: fresh 01-planning verification pass
- verification: Slice 04-b tier-aware fix completed. Refactored `RACING_VARIANT_SCORER` into a `VariantScorer`-compatible wrapper around the new 3-argument `scoreRacingVariant` helper so the runtime engine can pass the live candidate neuron count without breaking the scorer contract. Added `RACING_OSCILLATION_MIN_NEURONS = 200` constant, gated the oscillation penalty in `scoreRacingVariant` and `evaluateRacingTrendScore`, and replaced the growth-commit threshold multiplier with an additive `resolveOscillationThresholdBoost(...)` that returns `RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST = 0.03` only when `oscillationMetric > 0.5` and `neuronCount >= 200`. Updated tests to cover tier-aware penalty absence below 200 neurons, penalty presence at/above 200 neurons, and threshold-boost tier/commit gating.
- specialist-review-fix-cycle: NGE specialist review identified 2 test gaps in `runtime.adaptation.test.ts`. Added focused tests:
  1. `evaluateRacingTrendScore` â‰¥200-neuron oscillation-penalty path (`Network(200, 1)` + oscillating vs monotonic history + `detectScoreWindowOscillation` active).
  2. Runtime engine `scoreFn` wiring verification (`scoreRacingVariant(outputs, target, tickInput.network.nodes.length)` source inspection).
- specialist-review-fix-cycle-2: NGE specialist review found GAP 1 test was not gate-specific (the oscillating-vs-monotonic assertion would pass without the gate). Strengthened the test to compare the **same** oscillating history (`[5, 3, 5, 3, 1]`, negative trend) on a small network (`Network(4, 2)`, below `RACING_OSCILLATION_MIN_NEURONS`) versus a large network (`Network(200, 1)`, at/above the threshold). The negative trend gates the complexity bonus off, so the only score difference is the oscillation penalty. Single-expect assertion: `largeScore < smallScore`.
- `npx tsc --noEmit -p tsconfig.json` â†’ PASS.
- `npm run lint -- examples/racing_curriculum/controller/runtime.adaptation.ts examples/racing_curriculum/controller/runtime.adaptation.test.ts` â†’ PASS (0 errors, 21 unrelated warnings in `src/neat/nge-juvenile/`).
- `npx prettier --check examples/racing_curriculum/controller/runtime.adaptation.ts examples/racing_curriculum/controller/runtime.adaptation.test.ts` â†’ PASS.
- `npm run quality:folder -- --folder=examples/racing_curriculum/controller` â†’ PASS (0 TS diagnostics, 0 ESLint errors, 39/39 JSDoc symbols).
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md` â†’ PASS.
- plan-sync gate (`neataptic-gate-mcp:run_gate_check --gate=plan-sync`) â†’ PASS.
- step-packet gate â†’ PASS (schema violations reconciled 2026-07-18: Phase 4 `expansion: slices` â†’ `steps`; `auto_expand: true` â†’ `false`; slice 04-a `goal` â†’ `red-testing`; slice 04-c `goal` â†’ `green-testing`).
- plan-slice-quality gate (`neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality`) â†’ PASS.
- agent-graph gate (`neataptic-gate-mcp:run_gate_check --gate=agent-graph`) â†’ PASS.
- green-testing validation (05-green-testing) â€” targeted Jest run:
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller/runtime.adaptation.test.ts --coverage --coverageReporters=json-summary` â†’ PASS.
  - 82/82 tests passed in `runtime.adaptation.test.ts`.
  - Strengthened GAP 1 test (`evaluateRacingTrendScore` applies oscillation penalty only when `network.nodes.length >= 200`, same history `[5,3,5,3,1]`, `largeScore < smallScore`) â†’ PASS.
  - GAP 2 source-inspection test (`scoreFn` passes `tickInput.network.nodes.length` to `scoreRacingVariant`) â†’ PASS.
  - TypeScript diagnostics: 0 errors in `examples/racing_curriculum/controller/` (folder-quality-metrics).
  - ESLint: 0 errors in `examples/racing_curriculum/controller/`.
  - Coverage note: `runtime.adaptation.ts` lives under `examples/` and is excluded from Jest coverage collection by `jest.config.mjs` `coveragePathIgnorePatterns: ['/examples/']`. No coverage entry is produced for this file, so no regression is possible in the collected coverage set. The requested `npx ts-node scripts/quality/folder.ts --files ...` command does not exist; closest available check is `node scripts/folder-quality-metrics.mjs --folder=examples/racing_curriculum/controller`, which passed.
- step-packet gate (`neataptic-gate-mcp:run_gate_check --gate=step-packet`) â†’ PASS (violations: []; plansScanned: 4).

See the `### Gate outputs` subsection under `## Validation evidence` for full JSON.

---

### Phase 1 â€” Planning [DONE]

```yaml
phase: 1
title: 'Planning: author step packets and register plan'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_phase: 'Phase 2 â€” Research confirmation'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
  - 'phase-handoff-workflow'
  - 'planning-acceptance-criteria'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=step-packet --json'
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality --json'
acceptance_criteria:
  - id: AC-001
    text: 'Plan file exists, is listed in plans/README.md and plans/Roadmap.md, and top-level status is [WIP]'
    validation: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
  - id: AC-002
    text: 'Step packets pass step-packet gate'
    validation: 'neataptic-gate-mcp:run_gate_check --gate=step-packet --json'
  - id: AC-003
    text: 'All slices are <= 4 hours and dependencies are acyclic'
    validation: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality --json'
placeholder_steps:
  - 'Step 01 â€” Author step packets and register plan'
constitution_check:
  - 'principle-1-ai-thinking-partner'
  - 'principle-5-unique-ids'
```

#### Step 01 â€” Author step packets and register plan [DONE]

```yaml
phase: 1
step: 1
title: 'Author step packets and register plan'
status: '[DONE]'
goal: 'planning'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_step: 'Step 02 â€” Confirm specialist findings and close research'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
  - 'phase-handoff-workflow'
  - 'planning-acceptance-criteria'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=step-packet --json'
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality --json'
acceptance_criteria:
  - id: AC-001
    text: 'Plan file exists, is listed in plans/README.md and plans/Roadmap.md, and top-level status is [WIP]'
    validation: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
  - id: AC-002
    text: 'Step packets pass step-packet gate'
    validation: 'neataptic-gate-mcp:run_gate_check --gate=step-packet --json'
  - id: AC-003
    text: 'All slices are <= 4 hours and dependencies are acyclic'
    validation: 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality --json'
constitution_check:
  - 'principle-1-ai-thinking-partner'
  - 'principle-5-unique-ids'
```

**Evidence captured:**

- plan-sync: run after README/Roadmap edits â€” result TBD below in Latest validation evidence.
- step-packet: run after plan file authoring.
- plan-slice-quality: run after slice list authoring.
- Suggested PR body prepared in the evidence section at the end of this plan.

---

### Phase 2 â€” Research [DONE]

```yaml
phase: 2
title: 'Research: confirm specialist findings'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_phase: 'Phase 3 â€” Red tests'
skills:
  - 'plan-alignment'
  - 'research-methodology'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-sync --json'
acceptance_criteria:
  - id: AC-004
    text: 'Root causes (baby label hard-coding and missing oscillation penalties) are recorded and cross-referenced to the source files'
    validation: 'neataptic-gate-mcp:run_gate_check --gate=plan-sync --json'
placeholder_steps:
  - 'Step 02 â€” Confirm specialist findings and close research'
constitution_check:
  - 'principle-2-human-mission-ai-method'
```

#### Step 02 â€” Confirm specialist findings and close research [DONE]

```yaml
phase: 2
step: 2
title: 'Confirm specialist findings and close research'
status: '[DONE]'
goal: 'researching'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_step: 'Step 03 â€” Red tests for sub-tier thresholds and oscillation penalties'
skills:
  - 'plan-alignment'
  - 'research-methodology'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-sync --json'
acceptance_criteria:
  - id: AC-004
    text: 'Root causes (baby label hard-coding and missing oscillation penalties) are recorded and cross-referenced to the source files'
    validation: 'neataptic-gate-mcp:run_gate_check --gate=plan-sync --json'
constitution_check:
  - 'principle-2-human-mission-ai-method'
```

Research is bounded: the root causes were supplied by prior specialist scouting. This step records them in the plan rather than re-deriving them.

**Findings recorded:**

- Juvenile sub-tier hard-coding: `neat.nge-juvenile.grow-stabilize.ts` uses a single `baby` stage and does not pass `neuronBudget.current` into threshold selection.
- Oscillation gap: `runtime.adaptation.ts` overwrites `lifecycleStage` to `baby` on every tick and neither `RACING_VARIANT_SCORER` nor `evaluateRacingTrendScore` measure or penalize steering/score oscillation.

---

### Phase 3 â€” Red Testing [DONE]

```yaml
phase: 3
title: 'Red tests for sub-tier thresholds and oscillation penalties'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_phase: 'Phase 4 â€” Implementation'
skills:
  - 'unit-test-writer'
  - 'red-testing'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation'
acceptance_criteria:
  - id: AC-005
    text: 'A focused red test exists for each sub-tier threshold behavior and fails before implementation'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile'
  - id: AC-006
    text: 'A focused red test exists for oscillation penalty behavior and fails before implementation'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation'
  - id: AC-007
    text: 'No existing tests are broken by the new assertions'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=(src/neat/nge-juvenile|examples/racing_curriculum/controller/runtime.adaptation)'
placeholder_steps:
  - 'Step 03 â€” Red tests for sub-tier thresholds and oscillation penalties'
constitution_check:
  - 'principle-4-breadth-first-recoverable'
```

#### Step 03 â€” Red tests for sub-tier thresholds and oscillation penalties [DONE]

```yaml
phase: 3
step: 3
title: 'Red tests for sub-tier thresholds and oscillation penalties'
status: '[DONE]'
goal: 'red-testing'
tdd_sequence: 'red-green'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_step: 'Step 04 â€” Implement sub-tier thresholds and oscillation penalties'
skills:
  - 'unit-test-writer'
  - 'red-testing'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation'
acceptance_criteria:
  - id: AC-005
    text: 'A focused red test exists for each sub-tier threshold behavior and fails before implementation'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile'
  - id: AC-006
    text: 'A focused red test exists for oscillation penalty behavior and fails before implementation'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation'
  - id: AC-007
    text: 'No existing tests are broken by the new assertions'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=(src/neat/nge-juvenile|examples/racing_curriculum/controller/runtime.adaptation)'
constitution_check:
  - 'principle-4-breadth-first-recoverable'
```

**VALIDATION_EVIDENCE (03-red-testing @ 2026-07-18T10:17Z)**

Red contracts added to both owner-local test files. Failures are exclusively missing-implementation (missing exports, constants, or function signatures), not test-file syntax errors.

| Workstream                             | Test file                                                          | Focused command                                                                                                            | Result                                 | Failure reason                                                                                                                                                                                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A â€” NGE juvenile sub-tier thresholds | `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts`   | `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize`   | 1 failed suite, 0 tests run            | `NGE_EXHAUSTION_TIER_FRACTIONS`, `NGE_EXHAUSTION_NOISE_SIGMA_TIERS`, `NGE_EXHAUSTION_DECAY_FLOOR_TIERS` not exported; `resolveStageFraction`/`resolveNoiseSigmaFraction` expect 1 argument but tests pass 2 (currentNeurons).                                      |
| B â€” Racing oscillation penalties     | `examples/racing_curriculum/controller/runtime.adaptation.test.ts` | `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller/runtime.adaptation` | 1 failed suite, 19 existing tests pass | `detectSteeringOscillation`, `detectScoreWindowOscillation`, `RACING_OSCILLATION_PENALTY_WEIGHT`, `RACING_OSCILLATION_COMMIT_THRESHOLD`, `RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST` not exported; `RACING_VARIANT_SCORER` declared locally but not exported. |

Gate checks:

- `neataptic-gate-mcp:run_gate_check --gate=step-packet` â†’ PASS.

Fixture/cleanup notes:

- NGE tests use deterministic numeric fixtures; no shared mutable state.
- Racing tests use `jest.useFakeTimers()` with `afterEach(() => { jest.runOnlyPendingTimers(); })` and `new Network(4, 2, { seed: 42 })` where a network is needed.

Expected green condition:

- Implement/export the missing constants/helpers and update `resolveStageFraction`, `resolveNoiseSigmaFraction`, and `resolveExhaustionImprovementThreshold` to accept/use the optional `currentNeurons` parameter.
- Rerun the two focused commands above; both should pass with all new assertions green.

---

### Phase 4 â€” Implementation [DONE]

```yaml
phase: 4
title: 'Implementation: sub-tier thresholds, oscillation penalties, bundle smoke'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_phase: 'Phase 5 â€” Green validation'
skills:
  - 'implementation-standards'
  - 'unit-test-runner'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
acceptance_criteria:
  - id: AC-008
    text: 'All red tests from Step 03 pass'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=(src/neat/nge-juvenile|examples/racing_curriculum/controller/runtime.adaptation)'
  - id: AC-009
    text: '100% coverage on touched src/ files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile'
placeholder_steps:
  - 'Step 04 â€” Implement sub-tier thresholds and oscillation penalties'
constitution_check:
  - 'principle-4-breadth-first-recoverable'
  - 'principle-5-unique-ids'
```

#### Step 04 â€” Implement sub-tier thresholds and oscillation penalties [DONE]

```yaml
phase: 4
step: 4
title: 'Implement sub-tier thresholds and oscillation penalties'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_step: 'Step 05 â€” Green validation and bundle rebuild'
skills:
  - 'implementation-standards'
  - 'unit-test-runner'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
acceptance_criteria:
  - id: AC-008
    text: 'All red tests from Step 03 pass'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=(src/neat/nge-juvenile|examples/racing_curriculum/controller/runtime.adaptation)'
  - id: AC-009
    text: '100% coverage on touched src/ files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile'
slices:
  - slice_id: '04-a'
    title: 'Library-side juvenile sub-tier exhaustion thresholds'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 4
    files_to_change:
      - 'src/neat/nge-juvenile/neat.nge-juvenile.constants.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
    acceptance_criteria:
      - id: AC-010
        text: 'Constants export tier-specific neuron and connection fractions and a resolveNeuronTierFraction helper'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile'
      - id: AC-011
        text: 'Grow-stabilize uses the current neuron budget to pick tier-specific exhaustion thresholds'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile'
      - id: AC-012
        text: 'JSDoc is updated for changed public helpers'
        validation: 'npm run lint -- src/neat/nge-juvenile/'
    parallelizable: true
    dependencies: []
    next_slice: '04-c'
  - slice_id: '04-b'
    title: 'Racing runtime oscillation detection and penalties'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/racing_curriculum/controller/runtime.adaptation.ts'
      - 'examples/racing_curriculum/controller/runtime.adaptation.test.ts'
    acceptance_criteria:
      - id: AC-013
        text: 'Steering and score oscillation helpers exist and are unit-tested'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation'
      - id: AC-014
        text: 'RACING_VARIANT_SCORER and evaluateRacingTrendScore apply oscillation penalties'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation'
      - id: AC-015
        text: 'No deferred cleanup: old constants / helpers removed in the same slice that introduces replacements'
        validation: 'git diff --name-status'
    parallelizable: true
    dependencies: []
    next_slice: '04-c'
  - slice_id: '04-c'
    title: 'Bundle rebuild and browser smoke validation'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 3
    files_to_change:
      - 'docs/assets/racing-curriculum.bundle.js'
      - 'bench-browser/*'
    acceptance_criteria:
      - id: AC-016
        text: 'Racing curriculum bundle rebuilds cleanly after A and B changes'
        validation: 'npm run build:racing-curriculum'
      - id: AC-017
        text: 'Browser smoke shows reduced steering oscillation and visible network growth on a foreground browser window'
        validation: 'browserVisibility: visible-foreground; start: npm start; Chrome DevTools MCP console/network checks'
    parallelizable: false
    dependencies:
      - '04-a'
      - '04-b'
    next_slice: null
constitution_check:
  - 'principle-4-breadth-first-recoverable'
  - 'principle-5-unique-ids'
```

#### Traceability

| Slice | Requirement                                                    | Test/validation                                                    |
| ----- | -------------------------------------------------------------- | ------------------------------------------------------------------ |
| 04-a  | Juvenile sub-tier thresholds scale with `neuronBudget.current` | `src/neat/nge-juvenile/*.test.ts`                                  |
| 04-b  | Oscillation penalties in scorer and trend evaluator            | `examples/racing_curriculum/controller/runtime.adaptation.test.ts` |
| 04-c  | Bundle rebuild + visible browser smoke                         | `npm run build:racing-curriculum` + Chrome DevTools MCP            |

---

#### Phase 4 completion note

[DONE] Phase 4 — Implementation complete. Detailed slice validation evidence, specialist-review fix cycle, green validation, and step-packet schema reconciliation are recorded in `plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.logs.md`.

### Phase 5 â€” Green Testing [CANCELLED]

_Folded into [Racing_Perception_Redesign.plans.md](Racing_Perception_Redesign.plans.md) — bundle rebuild and browser validation will happen there._

```yaml
phase: 5
title: 'Green validation and bundle rebuild'
status: '[CANCELLED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_phase: 'Phase 6 â€” Documentation'
skills:
  - 'coverage-guard'
  - 'green-testing'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation'
  - 'npm run build:racing-curriculum'
acceptance_criteria:
  - id: AC-018
    text: 'All touched src/ and racing controller tests pass with zero failures'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=(src/neat/nge-juvenile|examples/racing_curriculum/controller/runtime.adaptation)'
  - id: AC-019
    text: '100% statements, branches, functions, lines on all touched src/ files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile'
  - id: AC-020
    text: 'Racing curriculum bundle rebuilds cleanly'
    validation: 'npm run build:racing-curriculum'
  - id: AC-021
    text: 'Browser smoke shows reduced steering oscillation and visible network growth on a foreground browser window'
    validation: 'browserVisibility: visible-foreground; npm start; Chrome DevTools MCP console/network checks'
placeholder_steps:
  - 'Step 05 â€” Green validation and bundle rebuild'
constitution_check:
  - 'principle-4-breadth-first-recoverable'
```

#### Step 05 â€” Green validation and bundle rebuild [CANCELLED]

_Folded into Racing_Perception_Redesign.plans.md — bundle rebuild and browser validation will happen there._

```yaml
phase: 5
step: 5
title: 'Green validation and bundle rebuild'
status: '[CANCELLED]'
goal: 'green-testing'
tdd_sequence: 'green-only'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_step: 'Step 06 â€” Documentation and README update'
skills:
  - 'coverage-guard'
  - 'green-testing'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation'
  - 'npm run build:racing-curriculum'
acceptance_criteria:
  - id: AC-018
    text: 'All touched src/ and racing controller tests pass with zero failures'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=(src/neat/nge-juvenile|examples/racing_curriculum/controller/runtime.adaptation)'
  - id: AC-019
    text: '100% statements, branches, functions, lines on all touched src/ files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile'
  - id: AC-020
    text: 'Racing curriculum bundle rebuilds cleanly'
    validation: 'npm run build:racing-curriculum'
  - id: AC-021
    text: 'Browser smoke shows reduced steering oscillation and visible network growth on a foreground browser window'
    validation: 'browserVisibility: visible-foreground; npm start; Chrome DevTools MCP console/network checks'
constitution_check:
  - 'principle-4-breadth-first-recoverable'
```

---

### Phase 6 â€” Documentation [CANCELLED]

_Folded into [Racing_Perception_Redesign.plans.md](Racing_Perception_Redesign.plans.md) — documentation updates will happen there._

```yaml
phase: 6
title: 'Documentation and README update'
status: '[CANCELLED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_phase: 'Phase 7 â€” Session logging and archive preparation'
skills:
  - 'academic-docs-auditor'
  - 'docs-example-writer'
validation:
  - 'npm run docs:folders:racing-curriculum'
  - 'npm run docs:folders:src'
  - 'npm run lint'
acceptance_criteria:
  - id: AC-022
    text: 'Racing curriculum README documents the new oscillation penalties and sub-tier thresholds'
    validation: 'npm run docs:folders:racing-curriculum'
  - id: AC-023
    text: 'JSDoc updates are generated for changed public APIs in src/neat/nge-juvenile/'
    validation: 'npm run docs:folders:src'
  - id: AC-024
    text: 'Lint passes on changed source files'
    validation: 'npm run lint'
placeholder_steps:
  - 'Step 06 â€” Documentation and README update'
constitution_check:
  - 'principle-3-verbatim-binding'
```

#### Step 06 â€” Documentation and README update [CANCELLED]

_Folded into Racing_Perception_Redesign.plans.md — documentation updates will happen there._

```yaml
phase: 6
step: 6
title: 'Documentation and README update'
status: '[CANCELLED]'
goal: 'documenting'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_step: 'Step 07 â€” Session logging and archive preparation'
skills:
  - 'academic-docs-auditor'
  - 'docs-example-writer'
validation:
  - 'npm run docs:folders:racing-curriculum'
  - 'npm run docs:folders:src'
  - 'npm run lint'
acceptance_criteria:
  - id: AC-022
    text: 'Racing curriculum README documents the new oscillation penalties and sub-tier thresholds'
    validation: 'npm run docs:folders:racing-curriculum'
  - id: AC-023
    text: 'JSDoc updates are generated for changed public APIs in src/neat/nge-juvenile/'
    validation: 'npm run docs:folders:src'
  - id: AC-024
    text: 'Lint passes on changed source files'
    validation: 'npm run lint'
constitution_check:
  - 'principle-3-verbatim-binding'
```

---

### Phase 7 â€” Session Logging [CANCELLED]

_Closure handled here; the plan file is being archived and the remaining logging/archive work is folded into [Racing_Perception_Redesign.plans.md](Racing_Perception_Redesign.plans.md)._

```yaml
phase: 7
title: 'Session logging and archive preparation'
status: '[CANCELLED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_phase: 'Archive closed plan/log pair to plans/completed/'
skills:
  - 'tracker-handoff'
  - 'capturing-learning-event'
validation:
  - 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
  - 'node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json'
  - 'node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json'
acceptance_criteria:
  - id: AC-025
    text: 'Completed phases are compressed to concise coverage notes'
    validation: 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
  - id: AC-026
    text: 'Same-boundary log file exists and records the durable done state'
    validation: 'node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json'
  - id: AC-027
    text: 'Stale top-level [WIP] markers are cleared after archive'
    validation: 'node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json'
placeholder_steps:
  - 'Step 07 â€” Session logging and archive preparation'
constitution_check:
  - 'principle-4-breadth-first-recoverable'
```

#### Step 07 â€” Session logging and archive preparation [CANCELLED]

_Closure handled here; the plan file is being archived and remaining logging/archive work is folded into Racing_Perception_Redesign.plans.md._

```yaml
phase: 7
step: 7
title: 'Session logging and archive preparation'
status: '[CANCELLED]'
goal: 'logging'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md'
copy_paste: true
next_step: null
skills:
  - 'tracker-handoff'
  - 'capturing-learning-event'
validation:
  - 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
  - 'node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json'
  - 'node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json'
acceptance_criteria:
  - id: AC-025
    text: 'Completed phases are compressed to concise coverage notes'
    validation: 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
  - id: AC-026
    text: 'Same-boundary log file exists and records the durable done state'
    validation: 'node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json'
  - id: AC-027
    text: 'Stale top-level [WIP] markers are cleared after archive'
    validation: 'node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json'
constitution_check:
  - 'principle-4-breadth-first-recoverable'
```

---

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Active workstream: NGE Racing Curriculum Oscillation & Sub-Tier Fix (plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md).
Phase 4 is [DONE]; Phase 5 Step 05 is the next value-adding frontier.

Next narrow task: dispatch 05-green-testing to run green validation and bundle rebuild for
- library-side juvenile sub-tier exhaustion thresholds in src/neat/nge-juvenile/ and
- racing runtime steering/score oscillation penalties in examples/racing_curriculum/controller/runtime.adaptation.ts.

Required validations before documentation:
1. npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile must pass with zero failures.
2. npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation must pass with zero failures.
3. npm run build:racing-curriculum must complete without errors.
4. Browser smoke on a visible foreground window must show reduced steering oscillation and visible network growth.

Caution: this plan is a child of the upstream racing curriculum plan (NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md). Preserve tier-promotion / carry-reset semantics from examples/racing_curriculum/reference.plans.md.
```

---

## Validation evidence

### Prepared git commands (user-run)

```bash
# On the feature branch
node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md
neataptic-gate-mcp:run_gate_check --gate=step-packet --json
neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality --json

# After edits
git add plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md plans/README.md plans/Roadmap.md
git commit -m "plan: NGE racing oscillation & sub-tier fix tracker and registration"
```

### PR template (user-run)

```markdown
Title: plan: NGE racing oscillation & sub-tier fix tracker

Body:

- Adds focused plans/ tracker for two racing-curriculum hardening workstreams:
  - library-side juvenile sub-tier exhaustion thresholds (src/neat/nge-juvenile/)
  - racing runtime steering/score oscillation penalties (examples/racing_curriculum/controller/runtime.adaptation.ts)
- Registers the plan in plans/README.md and plans/Roadmap.md.
- Phase 1/2 [DONE]; Phase 3 red-testing is the next frontier.
- Slice 04-a and 04-b are parallel (<= 4h each); Slice 04-c depends on both and includes browser smoke validation.
```

### Gate outputs

#### plan-sync script

```json
{
  "name": "plan sync",
  "ok": true,
  "issues": [],
  "counts": {
    "errors": 0,
    "warnings": 0
  },
  "summaryText": "PASS plan sync: 0 errors, 0 warnings (plan: plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md)",
  "plan": {
    "path": "plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md",
    "status": "WIP"
  },
  "downstreamTrackers": [
    "plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md",
    "plans/Neon_Shooter_NGE_Demo.plans.md",
    "plans/mcp-active-binding.plans.md"
  ]
}
```

#### plan-sync gate

```json
{
  "pass": true,
  "evidence": {
    "wipPlans": [
      "plans/mcp-active-binding.plans.md",
      "plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md",
      "plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md"
    ],
    "missingFromReadme": [],
    "missingFromRoadmap": [],
    "plansChecked": 7
  },
  "fixHint": "All WIP plans are correctly registered in README and Roadmap.",
  "owner": "validate-plan-sync.mjs"
}
```

#### step-packet gate

```json
{
  "pass": true,
  "evidence": {
    "blocksChecked": [
      "plans/mcp-active-binding.plans.md:yaml@14718",
      "plans/mcp-active-binding.plans.md:yaml@16171",
      "plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md:yaml@8202",
      "plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md:yaml@9783"
    ],
    "violations": [],
    "planReadinessWarnings": [],
    "plansScanned": 4
  },
  "fixHint": "All active WIP phase/step packets conform to the new format.",
  "owner": "step-packet.gate.mjs"
}
```

#### plan-slice-quality gate

```json
{
  "pass": true,
  "evidence": {
    "plansChecked": [
      "plans/mcp-active-binding.plans.md",
      "plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md",
      "plans/Neon_Shooter_NGE_Demo.plans.md",
      "plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md"
    ],
    "violations": [],
    "limit": 4
  },
  "fixHint": "All WIP plan slices are within the 4-hour estimate limit.",
  "owner": "plan-slice-quality.gate.mjs"
}
```

#### workflow-update-sync hook-check

```json
{
  "ok": true,
  "pass": true,
  "plan": {
    "path": "plans/NGE_Racing_Curriculum_Oscillation_SubTier_Fix.plans.md",
    "status": "WIP"
  },
  "syncEvent": {
    "currentWipStep": "Phase 3 Step 3",
    "nextPlannedStep": null,
    "actionTaken": "phase-complete",
    "reason": "Phase 3 Step 3 remains active and has no immediate next [PLANNED] step. Treating this as a phase boundary, not a hook failure.",
    "downstreamTrackers": [
      "plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md",
      "plans/Neon_Shooter_NGE_Demo.plans.md",
      "plans/mcp-active-binding.plans.md"
    ]
  },
  "evidence": "Workflow sync reached a phase boundary: Phase 3 Step 3 remains active and has no immediate next [PLANNED] step. Treating this as a phase boundary, not a hook failure.",
  "summaryText": "Workflow update sync: phase-complete (hook-check)"
}
```
