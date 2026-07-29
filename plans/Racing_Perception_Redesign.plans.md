# Racing Perception Redesign

**Status:** [PLANNED]

## Scope

Redesign the racing curriculum observation/perception layer to support
opponent-aware opponents. This plan covers three candidate approaches; only
Plans A and B are in scope now. Plan C is cancelled.

- **Plan A — Speed Bug Fix (standalone):** Persist car velocity in `CarState`
  and pipe real speed into the existing teammate slot. No input dimension change.
- **Plan B — Tier 6 Opponent Perception:** Add a new observation tier with
  103 base channels + 21 opponent channels (3 opponent slots × 7 ego-relative
  channels) = 124 total inputs.
- **Plan C — Track-Centric Frenét Refactor (CANCELLED):** Not needed —
  ego-relative + existing track-edge channels provide sufficient spatial awareness.

## Current state

Plan paused; active workstream is now `plans/Neon_Shooter_NGE_Demo.plans.md`.

Previous claim: 04-implementing @ 2026-07-18T16:30:00-04:00. Pre-green fix slice
`02-impl-tier6-pre-green-fixes` is implemented across the Tier 6
opponent-perception seam, the worker race-pack/evolution wrapper, the
coevolution JSDoc, and the browser-entry/network-view label set. All ten fix
items (teammate filtering, `RaceControllerNetwork.input` exposure, stale-speed
zeroing, body-frame speed projection, deterministic 9-output browser networks,
input-layer shrinking on tier downgrade, tier-aware 70/77/91/103/124 label
sets, richer I/O tooltips, updated Tier 6 JSDoc, and a 3v3 mixed-team fixture
with a non-zero heading speed-projection test) are in place.
Preflight (`tsc`, `eslint`, `prettier`, `quality:folder`, `npm run docs`) passed.
Green testing is pending for `05-green-testing` when this plan resumes.

## Source of truth

- Consolidated proposal:
  `C:/Users/reice/.copilot/session-state/a9ac2d1a-1970-4e78-8e0d-d4d34789d8f1/files/perception-redesign-proposal.md`

---

### Phase 1 — Speed Bug Fix [PLANNED]

**Phase objective:** Fix the speed persistence bug so the teammate observation
slot reflects actual car speed, without changing input dimensions or breaking
existing tiers.

```yaml
phase: 1
title: 'Speed Bug Fix'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Racing_Perception_Redesign.plans.md'
copy_paste: true
next_phase: 'Step 01 — Phase 2 red tests'
skills:
  - 'plan-alignment'
  - 'implementation-standards'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
acceptance_criteria:
  - id: AC-001
    text: 'CarState persists speedWorld, forwardSpeedWorld, and lateralSpeedWorld'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=environment.step'
  - id: AC-002
    text: 'Teammate observation slot uses real speed instead of hardcoded 0'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler'
  - id: AC-003
    text: 'No observation tier changes input dimensions; all existing Tier 1-5 tests still pass'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler'
constitution_check:
  - 'principle-4-small-slices'
placeholder_steps:
  - 'Step 01 — Implement speed bug fix'
```

#### Step 01: Implement speed bug fix [PLANNED]

```yaml
phase: 1
step: 1
title: 'Implement speed bug fix'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Racing_Perception_Redesign.plans.md'
copy_paste: true
next_step: 'Phase 2 Step 01 — Tier 6 red tests'
skills:
  - 'implementation-standards'
  - 'green-validation-gates'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=environment.step'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler'
  - 'npm run lint'
acceptance_criteria:
  - id: AC-001
    text: 'CarState persists speedWorld, forwardSpeedWorld, and lateralSpeedWorld'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=environment.step'
  - id: AC-002
    text: 'Teammate observation slot uses real speed instead of hardcoded 0'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler'
  - id: AC-003
    text: 'No observation tier changes input dimensions; all existing Tier 1-5 tests still pass'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler'
constitution_check:
  - 'principle-4-small-slices'
slices:
  - slice_id: '01-impl-speed-fields'
    title: 'Persist velocity fields and pipe speed to teammate slot'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/racing_curriculum/environment/environment.types.ts'
      - 'examples/racing_curriculum/environment/environment.step.service.ts'
      - 'examples/racing_curriculum/controller/observation.assembler.ts'
    acceptance_criteria:
      - id: AC-001
        text: 'CarState/RacingCarState exposes speedWorld, forwardSpeedWorld, lateralSpeedWorld'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=environment.step'
      - id: AC-002
        text: 'stepCarKinematics populates all three velocity fields'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=environment.step'
      - id: AC-003
        text: 'buildTeammateSlot reads teammate.speedWorld instead of hardcoding 0'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler'
    parallelizable: false
    dependencies: []
    next_slice: '01-green-speed-fix'
  - slice_id: '01-green-speed-fix'
    title: 'Green validation for speed fix'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'coverage/lcov.info'
    acceptance_criteria:
      - id: AC-004
        text: 'Focused environment and observation suites pass'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern="environment.step|observation.assembler"'
      - id: AC-005
        text: '100% coverage on touched src/ files'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern="environment.step|observation.assembler"'
      - id: AC-006
        text: 'Lint passes'
        validation: 'npm run lint'
    parallelizable: false
    dependencies:
      - '01-impl-speed-fields'
    next_slice: null
```

## PlanUpdate: 01-impl-speed-fields

```yaml
PlanUpdate:
  slice_id: '01-impl-speed-fields'
  changed_files:
    - 'examples/racing_curriculum/environment/environment.types.ts'
    - 'examples/racing_curriculum/environment/environment.step.service.ts'
    - 'examples/racing_curriculum/controller/observation.assembler.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check examples/racing_curriculum/environment/environment.types.ts examples/racing_curriculum/environment/environment.step.service.ts examples/racing_curriculum/controller/observation.assembler.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=environment.step'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler'
  rollback:
    - 'git checkout -- examples/racing_curriculum/environment/environment.types.ts'
    - 'git checkout -- examples/racing_curriculum/environment/environment.step.service.ts'
    - 'git checkout -- examples/racing_curriculum/controller/observation.assembler.ts'
  next: 'Run 05-green-testing for slice 01-green-speed-fix and attach coverage-guard evidence'
```

---

### Phase 2 — Tier 6 Opponent Perception [PLANNED]

**Phase objective:** Add a new observation tier (Tier 6) with 124 inputs:
103 base channels + 21 opponent channels from 3 ego-relative opponent slots.

```yaml
phase: 2
title: 'Tier 6 Opponent Perception'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Racing_Perception_Redesign.plans.md'
copy_paste: true
next_phase: 'Archive / Plan C deferred'
skills:
  - 'plan-alignment'
  - 'implementation-standards'
  - 'red-test-contracts'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality'
acceptance_criteria:
  - id: AC-010
    text: 'Tier 6 observation has exactly 124 channels'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler.tier6'
  - id: AC-011
    text: 'Each opponent slot exposes 7 ego-relative channels'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler.tier6'
  - id: AC-012
    text: 'Tier 6 coevolution branch uses 124 inputs and 9 outputs'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=simulation-worker.coevolution'
  - id: AC-013
    text: 'resolveControllerInputCountForObservationTier(6) returns 124'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=browser-entry'
  - id: AC-014
    text: 'Tier 1-5 byte layouts remain unchanged'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
placeholder_steps:
  - 'Step 01 — Tier 6 red tests'
  - 'Step 02 — Tier 6 implementation'
  - 'Step 03 — Tier 6 green validation'
```

#### Step 01: Tier 6 red tests [PLANNED]

```yaml
phase: 2
step: 1
title: 'Tier 6 red tests'
status: '[PLANNED]'
goal: 'red-testing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Racing_Perception_Redesign.plans.md'
copy_paste: true
next_step: 'Phase 2 Step 02 — Tier 6 implementation'
skills:
  - 'red-test-contracts'
  - 'planning-acceptance-criteria'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=observation.assembler.tier6'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.coevolution'
  - "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/browser-entry.test.ts'"
acceptance_criteria:
  - id: AC-015
    text: 'Red tests exist for Tier 6 observation length and opponent slot layout'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=observation.assembler.tier6'
  - id: AC-016
    text: 'Red tests exist for Tier 6 coevolution branch dimensions'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.coevolution'
  - id: AC-017
    text: 'Red tests exist for browser-entry Tier 6 input count resolution'
    validation: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/browser-entry.test.ts'"
constitution_check:
  - 'principle-4-small-slices'
slices:
  - slice_id: '02-red-tier6-observation'
    title: 'Write red tests for Tier 6 observation assembler'
    status: '[RED-COMPLETE]'
    goal: 'red-testing'
    estimate_hours: 3
    files_to_change:
      - 'examples/racing_curriculum/controller/observation.assembler.tier6.test.ts'
    acceptance_criteria:
      - id: AC-018
        text: 'Test expects 124 total channels and fails before Tier 6 is implemented'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=observation.assembler.tier6'
      - id: AC-019
        text: 'Test expects 3 opponent slots with 7 ego-relative channels each'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=observation.assembler.tier6'
      - id: AC-020
        text: 'Test expects zero-padding when fewer than 3 opponents are present'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=observation.assembler.tier6'
    parallelizable: false
    dependencies: []
    next_slice: '02-red-tier6-coevolution'
  - slice_id: '02-red-tier6-coevolution'
    title: 'Write red tests for Tier 6 coevolution branch'
    status: '[RED-COMPLETE]'
    goal: 'red-testing'
    estimate_hours: 3
    files_to_change:
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.test.ts'
    acceptance_criteria:
      - id: AC-021
        text: 'Test expects Tier 6 branch to create genomes with 124 inputs and 9 outputs'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.coevolution'
    parallelizable: false
    dependencies:
      - '02-red-tier6-observation'
    next_slice: '02-red-tier6-browser-entry'
  - slice_id: '02-red-tier6-browser-entry'
    title: 'Write red tests for browser-entry Tier 6 input count'
    status: '[RED-COMPLETE]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/browser-entry/browser-entry.test.ts'
    acceptance_criteria:
      - id: AC-022
        text: 'Test expects resolveControllerInputCountForObservationTier(6) === 124'
        validation: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/browser-entry.test.ts'"
      - id: AC-023
        text: 'Test expects SupportedObservationTier to include 6'
        validation: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/browser-entry.test.ts'"
    parallelizable: false
    dependencies:
      - '02-red-tier6-coevolution'
    next_slice: null
```

## PlanUpdate: 02-red-tier6-observation

```yaml
PlanUpdate:
  slice_id: '02-red-tier6-observation'
  changed_files:
    - 'examples/racing_curriculum/controller/observation.assembler.tier6.test.ts'
  preflight:
    - 'npx eslint examples/racing_curriculum/controller/observation.assembler.tier6.test.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=observation.assembler.tier6'
  red_evidence:
    command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=observation.assembler.tier6'
    exit_code: 1
    failing_tests: 8
    failing_reason: 'Missing Tier 6 observation export: assembleTier6Observation'
    compile_note: |
      Full `npx tsc --noEmit --skipLibCheck -p tsconfig.test.json` is blocked
      by an unrelated parse error in node_modules/devtools-protocol/types/protocol-mapping.d.ts.
      Focused Jest runs confirm the new test file compiles and runs.
  fixture_notes: |
    Static 6-car EnvironmentState with deterministic on-spline positions.
    Focal car at sample 0 heading 0; opponents placed at known offsets.
    Seeded RNG not required because assembler tests use fixed geometry.
  expected_green: |
    assembleTier6Observation exported from observation.assembler.ts returns
    a 124-channel Float32Array: 103 base channels + 3 opponent slots × 7
    ego-relative channels. Missing opponents zero-padded; roster order stable.
  rollback:
    - 'git checkout -- examples/racing_curriculum/controller/observation.assembler.tier6.test.ts'
  next: 'Run 04-implementing for slice 02-impl-tier6-observation'
```

## PlanUpdate: 02-red-tier6-coevolution

```yaml
PlanUpdate:
  slice_id: '02-red-tier6-coevolution'
  changed_files:
    - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.test.ts'
  preflight:
    - 'npx eslint examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.test.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.coevolution'
  red_evidence:
    command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.coevolution'
    exit_code: 1
    failing_tests: 1
    failing_reason: 'Expected 124 inputs, received 103'
    compile_note: |
      Full `npx tsc --noEmit --skipLibCheck -p tsconfig.test.json` is blocked
      by an unrelated parse error in node_modules/devtools-protocol/types/protocol-mapping.d.ts.
      Focused Jest runs confirm the modified test file compiles and runs.
  fixture_notes: |
    createCoevolutionContainer({ populationSize: 10, rngSeed: 42, tier: 6 }).
    First car genome inspected for input/output dimensions.
  expected_green: |
    simulation-worker.coevolution.service.ts builds Tier 6 genomes with
    124 inputs (103 base + 21 opponent perception) and 9 outputs.
  rollback:
    - 'git checkout -- examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.test.ts'
  next: 'Run 04-implementing for slice 02-impl-tier6-coevolution'
```

## PlanUpdate: 02-red-tier6-browser-entry

```yaml
PlanUpdate:
  slice_id: '02-red-tier6-browser-entry'
  changed_files:
    - 'examples/racing_curriculum/browser-entry/browser-entry.test.ts'
    - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
  preflight:
    - 'npx eslint examples/racing_curriculum/browser-entry/browser-entry.test.ts examples/racing_curriculum/browser-entry/browser-entry.ts'
  tests_for_green:
    - "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/browser-entry.test.ts'"
  red_evidence:
    command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/browser-entry.test.ts'"
    exit_code: 1
    failing_tests: 2
    failing_reason: |
      resolveControllerInputCountForObservationTier function body does not
      contain '124'; SupportedObservationTier type body does not contain '6'.
      Pre-existing TS errors in browser-entry.ts were fixed to make the test
      runnable (imported AccelerationMode; asserted parallelVariantCount).
    compile_note: |
      Full `npx tsc --noEmit --skipLibCheck -p tsconfig.test.json` is blocked
      by an unrelated parse error in node_modules/devtools-protocol/types/protocol-mapping.d.ts.
      Focused Jest runs confirm the modified source and test files compile and run.
  fixture_notes: |
    Source-text assertions read browser-entry.ts directly. No runtime fixture.
  expected_green: |
    browser-entry.ts types SupportedObservationTier as 1|2|3|4|5|6 and
    resolveControllerInputCountForObservationTier returns 124 for tier 6.
  rollback:
    - 'git checkout -- examples/racing_curriculum/browser-entry/browser-entry.test.ts'
    - 'git checkout -- examples/racing_curriculum/browser-entry/browser-entry.ts'
  next: 'Run 04-implementing for slice 02-impl-tier6-browser-entry'
```

#### Step 02: Tier 6 implementation [PLANNED]

```yaml
phase: 2
step: 2
title: 'Tier 6 implementation'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Racing_Perception_Redesign.plans.md'
copy_paste: true
next_step: 'Phase 2 Step 03 — Tier 6 green validation'
skills:
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler.tier6'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=simulation-worker.coevolution'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=browser-entry'
  - 'npm run lint'
acceptance_criteria:
  - id: AC-024
    text: 'assembleTier6Observation returns a 124-channel Float32Array'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler.tier6'
  - id: AC-025
    text: 'buildOpponentSlot computes ego-relative geometry and relative speeds'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler.tier6'
  - id: AC-026
    text: 'Tier 6 coevolution branch instantiates networks with 124 inputs and 9 outputs'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=simulation-worker.coevolution'
  - id: AC-027
    text: 'Browser entry resolves Tier 6 to 124 inputs'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=browser-entry'
  - id: AC-028
    text: 'Tier 1-5 byte layouts and dimensions remain unchanged'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: '02-impl-tier6-observation'
    title: 'Implement assembleTier6Observation and opponent slot helper'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/racing_curriculum/controller/observation.assembler.ts'
    acceptance_criteria:
      - id: AC-029
        text: 'ObservationTier type includes 6'
        validation: 'npx tsc --noEmit -p tsconfig.json'
      - id: AC-030
        text: 'assembleTier6Observation returns 124 channels'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler.tier6'
      - id: AC-031
        text: 'buildOpponentSlot produces 7 ego-relative channels'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler.tier6'
      - id: AC-032
        text: 'Missing opponents are zero-padded deterministically'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler.tier6'
    parallelizable: false
    dependencies:
      - '02-red-tier6-browser-entry'
    next_slice: '02-impl-tier6-coevolution'
  - slice_id: '02-impl-tier6-coevolution'
    title: 'Implement Tier 6 coevolution branch and worker types'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts'
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.types.ts'
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier5.ts'
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
    acceptance_criteria:
      - id: AC-033
        text: 'Tier 6 branch creates 6 genomes with 124 inputs and 9 outputs'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=simulation-worker.coevolution'
      - id: AC-034
        text: 'Race pack service routes Tier 6 to assembleTier6Observation'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=simulation-worker.coevolution'
    parallelizable: false
    dependencies:
      - '02-impl-tier6-observation'
    next_slice: '02-impl-tier6-browser-ui'
  - slice_id: '02-impl-tier6-browser-ui'
    title: 'Wire Tier 6 into browser entry and network view labels'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
      - 'examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts'
      - 'examples/racing_curriculum/controller/nge.controller.ts'
    acceptance_criteria:
      - id: AC-035
        text: 'SupportedObservationTier includes 6'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=browser-entry'
      - id: AC-036
        text: 'resolveControllerInputCountForObservationTier(6) === 124'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=browser-entry'
      - id: AC-037
        text: 'Network view constants label the 124 input groups'
        validation: 'npm run lint'
    parallelizable: false
    dependencies:
      - '02-impl-tier6-coevolution'
    next_slice: null
```

## PlanUpdate: 02-impl-tier6-coevolution

```yaml
PlanUpdate:
  slice_id: '02-impl-tier6-coevolution'
  changed_files:
    - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts'
    - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.types.ts'
    - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier5.ts'
    - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx eslint examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.types.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier5.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
    - 'npx prettier --check examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.types.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier5.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
    - 'npm run quality:folder -- --folder=examples/racing_curriculum/workers/simulation-worker'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=simulation-worker.coevolution'
  rollback:
    - 'git checkout -- examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts'
    - 'git checkout -- examples/racing_curriculum/workers/simulation-worker/simulation-worker.types.ts'
    - 'git checkout -- examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier5.ts'
    - 'git checkout -- examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
  next: 'Run 05-green-testing with the focused coevolution test slice and attach coverage-guard evidence for the four changed files.'
```

VALIDATION_EVIDENCE:

- tsc: pass (npx tsc --noEmit -p tsconfig.json)
- eslint: 0 errors on changed files
- prettier: all changed files formatted
- quality:folder: PASS for examples/racing_curriculum/workers/simulation-worker
- tests: NOT RUN (owned by 05-green-testing)

## PlanUpdate: 02-impl-tier6-observation

```yaml
PlanUpdate:
  slice_id: '02-impl-tier6-observation'
  changed_files:
    - 'examples/racing_curriculum/controller/observation.assembler.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx eslint examples/racing_curriculum/controller/observation.assembler.ts'
    - 'npx prettier --check examples/racing_curriculum/controller/observation.assembler.ts'
    - 'npm run quality:folder -- --folder=examples/racing_curriculum/controller'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler.tier6'
  rollback:
    - 'git checkout -- examples/racing_curriculum/controller/observation.assembler.ts'
  next: 'Run 05-green-testing on observation.assembler.tier6.test.ts and attach coverage-guard evidence'
```

## PlanUpdate: 02-impl-tier6-browser-ui

```yaml
PlanUpdate:
  slice_id: '02-impl-tier6-browser-ui'
  changed_files:
    - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
    - 'examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts'
    - 'examples/racing_curriculum/browser-entry/network-view/network-view.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx eslint examples/racing_curriculum/browser-entry/browser-entry.ts examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts examples/racing_curriculum/browser-entry/network-view/network-view.ts'
    - 'npx prettier --check examples/racing_curriculum/browser-entry/browser-entry.ts examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts examples/racing_curriculum/browser-entry/network-view/network-view.ts'
    - 'npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry'
  tests_for_green:
    - "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/browser-entry.test.ts'"
    - "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/network-view/network-view.test.ts'"
  rollback:
    - 'git checkout -- examples/racing_curriculum/browser-entry/browser-entry.ts'
    - 'git checkout -- examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts'
    - 'git checkout -- examples/racing_curriculum/browser-entry/network-view/network-view.ts'
  next: 'Run 05-green-testing on the browser-entry and network-view focused test slices and attach coverage-guard evidence for the three changed files.'
```

VALIDATION_EVIDENCE:

- tsc: pass (npx tsc --noEmit -p tsconfig.json)
- eslint: 0 errors on changed files
- prettier: all changed files formatted
- quality:folder: PASS for examples/racing_curriculum/browser-entry
- plan-sync: pass (neataptic-gate-mcp-run_gate_check plan-sync)
- tests: NOT RUN (owned by 05-green-testing)

PR & review commands (run locally by user; do not push automatically):

```text
Branch: implement/tier6-browser-ui-02-impl

git checkout -b implement/tier6-browser-ui-02-impl
git add examples/racing_curriculum/browser-entry/browser-entry.ts examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts examples/racing_curriculum/browser-entry/network-view/network-view.ts
git commit -m "Wire Tier 6 opponent perception into racing browser UI and network-view labels — PlanUpdate: plans/Racing_Perception_Redesign.plans.md"
git push origin implement/tier6-browser-ui-02-impl
```

Then open a PR and paste the resulting `pr_url` into this section:

`pr_url:` ______________________________________________

HandoffPayload:

```json
{
  "slice_id": "02-impl-tier6-browser-ui",
  "changed_files": [
    "examples/racing_curriculum/browser-entry/browser-entry.ts",
    "examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts",
    "examples/racing_curriculum/browser-entry/network-view/network-view.ts"
  ],
  "preflight_outputs": {
    "tsc": "pass (npx tsc --noEmit -p tsconfig.json)",
    "eslint": "0 errors on changed files",
    "prettier": "all changed files formatted",
    "quality:folder": "PASS for examples/racing_curriculum/browser-entry",
    "plan-sync": "pass (neataptic-gate-mcp-run_gate_check plan-sync)"
  },
  "tests_for_green": [
    "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/browser-entry.test.ts'",
    "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/network-view/network-view.test.ts'"
  ],
  "coverage_guard_files": [
    "examples/racing_curriculum/browser-entry/browser-entry.ts",
    "examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts",
    "examples/racing_curriculum/browser-entry/network-view/network-view.ts"
  ],
  "pr_url": ""
}
```

## PlanUpdate: 02-impl-tier6-pre-green-fixes

```yaml
PlanUpdate:
  slice_id: '02-impl-tier6-pre-green-fixes'
  changed_files:
    - 'examples/racing_curriculum/controller/observation.assembler.ts'
    - 'examples/racing_curriculum/controller/observation.assembler.tier6.test.ts'
    - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
    - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.evolution.protocol.service.ts'
    - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts'
    - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
    - 'examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts'
    - 'examples/racing_curriculum/browser-entry/network-view/README.md'
    - 'examples/racing_curriculum/controller/README.md'
    - 'examples/racing_curriculum/workers/simulation-worker/README.md'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check examples/racing_curriculum/controller/observation.assembler.ts examples/racing_curriculum/controller/observation.assembler.tier6.test.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.evolution.protocol.service.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts examples/racing_curriculum/browser-entry/browser-entry.ts examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts'
    - 'npm run quality:folder -- --folder=examples/racing_curriculum/controller'
    - 'npm run quality:folder -- --folder=examples/racing_curriculum/workers/simulation-worker'
    - 'npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry'
    - 'npm run docs'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler.tier6'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=simulation-worker.coevolution'
    - "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/browser-entry.test.ts'"
    - "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/network-view/network-view.test.ts'"
  rollback:
    - 'git checkout -- examples/racing_curriculum/controller/observation.assembler.ts'
    - 'git checkout -- examples/racing_curriculum/controller/observation.assembler.tier6.test.ts'
    - 'git checkout -- examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
    - 'git checkout -- examples/racing_curriculum/workers/simulation-worker/simulation-worker.evolution.protocol.service.ts'
    - 'git checkout -- examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts'
    - 'git checkout -- examples/racing_curriculum/browser-entry/browser-entry.ts'
    - 'git checkout -- examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts'
  next: 'Run 05-green-testing on the focused Tier 6 slices and attach coverage-guard evidence for the seven source files.'
```

VALIDATION_EVIDENCE:

- tsc: pass (npx tsc --noEmit -p tsconfig.json)
- eslint/lint: 0 errors on changed files (21 pre-existing unrelated `any` warnings in src/neat/nge-juvenile)
- prettier: all changed source files formatted
- quality:folder: PASS for examples/racing_curriculum/controller, workers/simulation-worker, and browser-entry
- docs: npm run docs completed (READMEs refreshed)
- plan-sync: pass (neataptic-gate-mcp:run_gate_check --gate=plan-sync)
- tests: NOT RUN (owned by 05-green-testing)

PR & review commands (run locally by user; do not push automatically):

```text
Branch: implement/tier6-pre-green-fixes

git checkout -b implement/tier6-pre-green-fixes
git add examples/racing_curriculum/controller/observation.assembler.ts examples/racing_curriculum/controller/observation.assembler.tier6.test.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.evolution.protocol.service.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts examples/racing_curriculum/browser-entry/browser-entry.ts examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts examples/racing_curriculum/browser-entry/network-view/README.md examples/racing_curriculum/controller/README.md examples/racing_curriculum/workers/simulation-worker/README.md
git commit -m "Tier 6 opponent-perception pre-green fixes: teammate filtering, body-frame speed projection, worker input exposure, browser label sets, and mixed-team tests — PlanUpdate: plans/Racing_Perception_Redesign.plans.md"
git push origin implement/tier6-pre-green-fixes
```

Then open a PR and paste the resulting `pr_url` into this section:

`pr_url:` ________________________________________________

HandoffPayload:

```json
{
  "slice_id": "02-impl-tier6-pre-green-fixes",
  "changed_files": [
    "examples/racing_curriculum/controller/observation.assembler.ts",
    "examples/racing_curriculum/controller/observation.assembler.tier6.test.ts",
    "examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts",
    "examples/racing_curriculum/workers/simulation-worker/simulation-worker.evolution.protocol.service.ts",
    "examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts",
    "examples/racing_curriculum/browser-entry/browser-entry.ts",
    "examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts"
  ],
  "preflight_outputs": {
    "tsc": "pass (npx tsc --noEmit -p tsconfig.json)",
    "eslint": "0 errors on changed files",
    "prettier": "all changed source files formatted",
    "quality:folder": "PASS for controller, simulation-worker, and browser-entry folders",
    "docs": "npm run docs completed",
    "plan-sync": "pass (neataptic-gate-mcp:run_gate_check --gate=plan-sync)"
  },
  "tests_for_green": [
    "npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler.tier6",
    "npx jest --config=jest.config.mjs --no-cache --testPathPattern=simulation-worker.coevolution",
    "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/browser-entry.test.ts'",
    "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/network-view/network-view.test.ts'"
  ],
  "coverage_guard_files": [
    "examples/racing_curriculum/controller/observation.assembler.ts",
    "examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts",
    "examples/racing_curriculum/workers/simulation-worker/simulation-worker.evolution.protocol.service.ts",
    "examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts",
    "examples/racing_curriculum/browser-entry/browser-entry.ts",
    "examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts"
  ],
  "pr_url": ""
}
```

#### Step 03: Tier 6 green validation [PLANNED]

```yaml
phase: 2
step: 3
title: 'Tier 6 green validation'
status: '[PLANNED]'
goal: 'green-testing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Racing_Perception_Redesign.plans.md'
copy_paste: true
next_step: 'Phase 2 Step 04 — Documentation / closure'
skills:
  - 'green-validation-gates'
  - 'coverage-guard'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler.tier6'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=simulation-worker.coevolution'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=browser-entry'
  - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern="observation.assembler.tier6|simulation-worker.coevolution|browser-entry"'
  - 'npm run lint'
acceptance_criteria:
  - id: AC-038
    text: 'All Tier 6 red tests now pass'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern="observation.assembler.tier6|simulation-worker.coevolution|browser-entry"'
  - id: AC-039
    text: '100% coverage on touched src/ files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern="observation.assembler.tier6|simulation-worker.coevolution|browser-entry"'
  - id: AC-040
    text: 'Tier 1-5 regression tests still pass'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler'
  - id: AC-041
    text: 'Lint passes'
    validation: 'npm run lint'
constitution_check:
  - 'principle-4-small-slices'
slices:
  - slice_id: '02-green-tier6-focused'
    title: 'Green validation of Tier 6 tests and coverage'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 3
    files_to_change:
      - 'coverage/lcov.info'
    acceptance_criteria:
      - id: AC-042
        text: 'Tier 6 observation, coevolution, and browser-entry tests pass'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern="observation.assembler.tier6|simulation-worker.coevolution|browser-entry"'
      - id: AC-043
        text: '100% coverage on touched src/ files'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern="observation.assembler.tier6|simulation-worker.coevolution|browser-entry"'
    parallelizable: false
    dependencies:
      - '02-impl-tier6-browser-ui'
    next_slice: '02-green-tier6-regression'
  - slice_id: '02-green-tier6-regression'
    title: 'Regression validation for Tier 1-5'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'coverage/lcov.info'
    acceptance_criteria:
      - id: AC-044
        text: 'Existing observation assembler tests for Tier 1-5 still pass'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=observation.assembler'
      - id: AC-045
        text: 'Lint passes'
        validation: 'npm run lint'
    parallelizable: false
    dependencies:
      - '02-green-tier6-focused'
    next_slice: null
```

---

## Plan C — CANCELLED: Not needed — ego-relative + existing track-edge channels provide sufficient spatial awareness.

**Status:** [CANCELLED]

**Rationale:** The ego-relative Tier 6 design plus the existing track-edge
channels already provides enough spatial awareness for opponent-aware racing.
The continuous arc-length projection prerequisite is no longer justified.

**Reactivation condition:** None planned; if future track-centric lane-relative
perception becomes necessary, a new plan should be authored from current state.

---

## Latest validation evidence

- **Verification pass timestamp:** 2026-07-18T14:08-04:00.
- **Path correction:** All `src/racing/...` paths were factual errors; they have been
  corrected to the actual `examples/racing_curriculum/...` locations. Verified that
  the corrected source files exist, except for
  `examples/racing_curriculum/controller/observation.assembler.tier6.test.ts`,
  which is the new red-test file this plan will create and is consistent with the
  existing `observation.assembler.tierN.test.ts` naming convention in that folder.
- `neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality` → pass
  (all slices ≤ 4 hours, no violations in Racing_Perception_Redesign).
- `neataptic-gate-mcp:run_gate_check --gate=step-packet` → pass
  (checked active WIP packets; plan is `[PLANNED]` so its packets were not scanned).
- **green-light: true** — path blocker resolved; slice sizes and packet structure
  conform to workflow requirements. Plan is ready for execution-phase dispatch
  once it is moved to `[WIP]` and registered in `plans/README.md` / `plans/Roadmap.md`.
- Slice `02-impl-tier6-observation` implementation preflight (`2026-07-18T19:20-04:00`):
  - `npx tsc --noEmit -p tsconfig.json` → pass
  - `npx eslint examples/racing_curriculum/controller/observation.assembler.ts` → 0 errors
  - `npx prettier --check examples/racing_curriculum/controller/observation.assembler.ts` → pass
  - `npm run quality:folder -- --folder=examples/racing_curriculum/controller` → pass
  - `neataptic-gate-mcp:run_gate_check --gate=plan-sync` → pass
  - `neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality` → pass
  - `neataptic-gate-mcp:run_gate_check --gate=step-packet` → pass; emits a
    plan-readiness warning about a missing green-light marker, but the marker
    above is present.
