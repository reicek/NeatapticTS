# CI Failure Hardening

**Status:** [WIP]

Claim: Reopened to add native-ESM Jest project for accurate `.mjs` gate-script coverage.

**Objective:** Make the CI suite pass on the GitHub Actions `ubuntu-latest`
runner where SQLite, GPU, Windows-specific binaries, and dynamically rebuilt
`docs/`/`dist/` folders are unavailable. Fix local-only test assumptions
without breaking local development, close the reported 100 % coverage
gaps on the three barrel/re-export files, and ensure the new
`scripts/agent-customization/` gate `.mjs` files report accurate coverage by
running them under a native-ESM Jest project with the V8 coverage provider.

## Scope

- `scripts/agent-customization/mcp/__tests__/lazy-facade.red.test.ts`
- `scripts/mcp-semantic/__tests__/repo-cortex-mcp.red.test.ts`
- `scripts/mcp-semantic/repo-cortex-mcp.test.ts`
- `scripts/agent-customization/hooks/runtime-enforcement-hooks.test.ts`
- `scripts/agent-customization/browser-tests/__tests__/scenario-parser.test.ts`
- `src/architecture/activationArrayPool.ts`
- `src/neat/neat.diversity.ts`
- `src/neat/neat.lineage.ts`
- `scripts/agent-customization/gates/code-coverage.gate.mjs` (coverage accuracy via native-ESM project)
- `scripts/agent-customization/gates/merge-coverage-summaries.mjs` (coverage accuracy via native-ESM project)
- `scripts/agent-customization/gates/step-packet.gate.mjs` (coverage accuracy via native-ESM project)
- `scripts/agent-customization/mcp/neataptic-gate-mcp.mjs` (coverage accuracy via native-ESM project)
- New native-ESM Jest project `agent-customization-mjs` in `jest.config.mjs`

## Non-goals

- No production code behavior changes outside test/fixture files.
- No new dynamic-folder dependencies.
- No deferred cleanup: every temporary path used by tests must be created or
  removed in the same test that uses it.

## Latest validation evidence

green-light: true

- `plan-sync` gate: **PASS** after reactivating the plan in `plans/README.md` and `plans/Roadmap.md`.
  - Command: `node scripts/agent-customization/gates/plan-sync.gate.mjs --json`
  - Result:
    ```json
    {
      "pass": true,
      "evidence": {
        "wipPlans": [
          "plans/CI_Failure_Hardening.plans.md",
          "plans/mcp-active-binding.plans.md"
        ],
        "missingFromReadme": [],
        "missingFromRoadmap": [],
        "plansChecked": 6
      },
      "fixHint": "All WIP plans are correctly registered in README and Roadmap.",
      "owner": "validate-plan-sync.mjs"
    }
    ```
- `plan-slice-quality` gate: **PASS**
  - Command: `node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json`
  - Result:
    ```json
    {
      "pass": true,
      "evidence": {
        "plansChecked": [
          "plans/CI_Failure_Hardening.plans.md",
          "plans/mcp-active-binding.plans.md",
          "plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md"
        ],
        "violations": [],
        "limit": 4
      },
      "fixHint": "All WIP plan slices are within the 4-hour estimate limit.",
      "owner": "plan-slice-quality.gate.mjs"
    }
    ```
- `step-packet` gate: **PASS**
  - Command: `node scripts/agent-customization/gates/step-packet.gate.mjs --json`
  - Result:
    ```json
    {
      "pass": true,
      "evidence": {
        "blocksChecked": [
          "plans/CI_Failure_Hardening.plans.md:yaml@23761",
          "plans/mcp-active-binding.plans.md:yaml@14616",
          "plans/mcp-active-binding.plans.md:yaml@16069"
        ],
        "violations": [],
        "planReadinessWarnings": [],
        "plansScanned": 3
      },
      "fixHint": "All active WIP phase/step packets conform to the new format.",
      "owner": "step-packet.gate.mjs"
    }
    ```

- Step 03 durable baseline generator and focused green validation: **PASS**.
  - Replaced the temporary `tmp/generate-coverage-baseline.mjs` with a durable `--baseline` mode in `scripts/agent-customization/gates/merge-coverage-summaries.mjs`.
  - `merge-coverage-summaries.mjs` and `merge-coverage-summaries.test.mjs` keep 100/100/100/100 coverage (36/36 native-ESM tests pass).
  - `node scripts/agent-customization/gates/merge-coverage-summaries.mjs --input coverage --output coverage/coverage-summary.json --baseline=coverage/coverage-baseline.json` merged 5 per-project summaries and generated the git-derived baseline.
  - `node scripts/agent-customization/gates/code-coverage.gate.mjs --json` → `pass: true` for 9 changed source files, 0 missing, 0 failed.
  - Focused native-ESM tests pass: `code-coverage.gate.test.mjs` 33/33, `merge-coverage-summaries.test.mjs` 36/36, `neataptic-gate-mcp.test.mjs` 21/21.
  - Preflight: `npx tsc --noEmit -p tsconfig.json` and `tsconfig.test.json` exit 0; `npm run lint` exit 0; Prettier check clean.

- Independent 01-planning verification — 2026-07-08: `green-light: true`.
  - Reopened `plans/CI_Failure_Hardening.plans.md` from `plans/completed/`.
  - Added Step 02b packet (native-ESM Jest project) with four slices (2, 3, 2, 1 hours; all ≤ 4 hours).
  - Updated Step 03 validation to include focused `agent-customization-mjs` runs and the merged `code-coverage` gate.
  - Reverted Step 04 to `[PLANNED]` and repointed `source_of_truth` and validation commands to `plans/`.
  - Updated `plans/README.md` and `plans/Roadmap.md` to list CI Failure Hardening as `[WIP]`.
  - Gate results: `plan-slice-quality` PASS, `step-packet` PASS, `plan-sync` PASS.
  - Recommended next agent: `04-implementing` on Step 02b slice `02b-config`.
- `workflow-update-sync` hook: **PASS** with action `between-steps`.
  - Command: `node .github/hooks/workflow-update-sync.mjs --plan=plans/CI_Failure_Hardening.plans.md --json`
  - Result:
    ```json
    {
      "ok": true,
      "pass": true,
      "plan": {
        "path": "plans/CI_Failure_Hardening.plans.md",
        "status": "WIP"
      },
      "syncEvent": {
        "currentWipStep": null,
        "nextPlannedStep": null,
        "actionTaken": "between-steps",
        "reason": "Phase 1 is [WIP] but no step is currently [WIP]; the plan is between steps and awaiting the next step to become active.",
        "downstreamTrackers": [
          "plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md",
          "plans/mcp-active-binding.plans.md"
        ]
      },
      "evidence": "Workflow sync between steps: Phase 1 is [WIP] but no step is currently [WIP]; the plan is between steps and awaiting the next step to become active.",
      "summaryText": "Workflow update sync: between-steps"
    }
    ```
  - Note: the hook only parses integer `#### Step NN` headers; the non-integer `Step 02b` marker is correctly tracked by the `step-packet` gate but is invisible to the auto-advance hook. This is acceptable because Step 02b is explicitly `[WIP]` in its YAML block and header.

## Clarifications

- Q: Should local-only integration tests be skipped or mocked on CI?
  A: Skip on CI via a `process.env.CI || process.env.GITHUB_ACTIONS` guard.
  Mocking the entire SQLite + embedding corpus is out of scope for this
  hardening pass.
- Q: Should the `docs/browser-tests/webgpu-inference-smoke.html` dependency be
  replaced by a committed fixture or an inline mock?
  A: Replace with a committed fixture at
  `testing/fixtures/webgpu-inference-smoke.html` and point the test at it.
  The fixture only needs the token strings the assertions check.
- Q: How should the Windows-only `resolveSpawnCommand` fallback tests be
  handled on Linux CI?
  A: Skip the two tests that require a real `npx.cmd` next to the Node
  executable on non-Windows platforms. Add a new fixture-based test that
  covers the `.exe` branch of the same function so `lazy-facade-core.mjs`
  line 412 reaches 100 % statements.
- Q: What about the coverage gaps on `activationArrayPool.ts`,
  `neat.diversity.ts`, and `neat.lineage.ts`?
  A: Add minimal barrel-import tests that exercise every re-export line.
  These files are public API surfaces, so they should be covered rather than
  excluded.

---

### Phase 1 — CI Failure Hardening [WIP]

**Phase objective:** Remove all CI assumptions about local SQLite, GPU,
Windows binaries, and dynamic `docs/`/`dist/` folders; close the reported
100 % coverage gaps.

**Phase progression rule:** Start with Step 01. Step 01 must author the
remaining numbered step packets before the phase can advance.

#### Step 01: Plan the CI hardening phase [DONE]

```yaml
phase: 1
step: 1
goal: 'planning'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/CI_Failure_Hardening.plans.md'
copy_paste: true
next_step: 'Step 02 — Harden local-resource tests and add barrel coverage tests'
skills:
  - 'plan-alignment'
  - 'planning-acceptance-criteria'
  - 'phase-handoff-workflow'
validation:
  - 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json'
  - 'node scripts/agent-customization/gates/step-packet.gate.mjs --json'
acceptance_criteria:
  - id: AC-001
    text: 'Plan contains a [WIP] Step 02 with five slices, each estimate <= 4 hours'
    validation: 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json'
  - id: AC-002
    text: 'All phase and step YAML blocks pass the step-packet gate'
    validation: 'node scripts/agent-customization/gates/step-packet.gate.mjs --json'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
```

**User instruction:** Paste this full step packet.

**Step objective:** Author a machine-readable plan that maps every CI failure
to a small implementation slice and records the chosen skip/mock/fixture
decisions.

**Context the agent must know:**

- CI environment: `ubuntu-latest`, Node 24, no SQLite, no GPU, no prebuilt
  `docs/` or `dist/`.
- `jest.config.mjs` defines the `agent-customization-scripts` and
  `mcp-semantic-scripts` projects.
- `npm run test:smoke` only selects the `starter-examples` project, but the
  full `npm test` matrix exercises every project and is the quality gate the
  user cares about.

**Execution steps:**

1. Read the failing test files, `jest.config.mjs`, and `.github/workflows/ci.yml`.
2. Record the chosen strategy (skip / mock / fixture) per failure group.
3. Define acceptance criteria with AC-### IDs and traceability tables.
4. Author Step 02-05 packets and a skipped documentation step packet.
5. Run `plan-slice-quality` and `step-packet` gates.
6. Record gate output in `## Latest validation evidence`.

**Stop conditions:**

- Done when the plan file exists, gates pass, and the next active step is
  clearly identified.
- Blocked if a failure group cannot be classified as skip/mock/fixture;
  escalate to `00.cross-tier-helper`.

**Required validation:**

```bash
node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json
node scripts/agent-customization/gates/step-packet.gate.mjs --json
```

**Plan update requirement:** Update the source plan with gate evidence before
ending.

---

#### Step 02: Harden local-resource tests and add barrel coverage tests [DONE]

```yaml
phase: 1
step: 2
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/CI_Failure_Hardening.plans.md'
copy_paste: true
next_step: 'Step 02b — Add native-ESM Jest project for agent-customization .mjs coverage'
skills:
  - 'implementation-standards'
  - 'green-validation-gates'
  - 'plan-sync-validation'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=lazy-facade.red.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects mcp-semantic-scripts --testPathPattern=repo-cortex-mcp'
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=runtime-enforcement-hooks.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=scenario-parser.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern="activationArrayPool|neat\\.diversity|neat\\.lineage"'
acceptance_criteria:
  - id: AC-003
    text: 'lazy-facade self-check test creates tmp/ before writing and passes on CI'
    validation: 'npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=lazy-facade.red.test.ts'
  - id: AC-004
    text: 'Windows-only resolveSpawnCommand fallback tests are skipped on non-Windows platforms'
    validation: 'npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=lazy-facade.red.test.ts'
  - id: AC-005
    text: 'resolveSpawnCommand .exe branch is covered and lazy-facade-core.mjs reaches 100 % statements'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --selectProjects agent-customization-scripts --testPathPattern=lazy-facade.red.test.ts'
  - id: AC-006
    text: 'repo-cortex integration tests are skipped on CI (no SQLite assumption)'
    validation: 'CI=true npx jest --config=jest.config.mjs --no-cache --selectProjects mcp-semantic-scripts --testPathPattern=repo-cortex-mcp'
  - id: AC-007
    text: 'runtime-enforcement-hooks tests are skipped on CI (no SQLite index refresh)'
    validation: 'CI=true npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=runtime-enforcement-hooks.test.ts'
  - id: AC-008
    text: 'scenario-parser.test.ts reads a committed fixture instead of docs/browser-tests/'
    validation: 'npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=scenario-parser.test.ts'
  - id: AC-009
    text: 'activationArrayPool.ts, neat.diversity.ts, and neat.lineage.ts reach 100 % statements'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern="activationArrayPool|neat\\.diversity|neat\\.lineage"'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: '02-lazy-facade'
    title: 'Fix lazy-facade CI assumptions and cover .exe branch'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'scripts/agent-customization/mcp/__tests__/lazy-facade.red.test.ts'
    acceptance_criteria:
      - id: AC-010
        text: 'Self-check invalid-snapshot test creates tmp/ directory before writing'
      - id: AC-011
        text: 'Windows-specific fallback tests are guarded by process.platform === win32'
      - id: AC-012
        text: 'New fixture test covers resolveSpawnCommand .exe branch'
    parallelizable: true
    dependencies: []
    next_slice: '02-repo-cortex'
  - slice_id: '02-repo-cortex'
    title: 'Skip repo-cortex integration tests on CI'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'scripts/mcp-semantic/__tests__/repo-cortex-mcp.red.test.ts'
      - 'scripts/mcp-semantic/repo-cortex-mcp.test.ts'
    acceptance_criteria:
      - id: AC-013
        text: 'Both repo-cortex describe blocks are skipped when CI env is set'
    parallelizable: true
    dependencies: []
    next_slice: '02-runtime-hooks'
  - slice_id: '02-runtime-hooks'
    title: 'Skip runtime-enforcement-hooks tests on CI'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - 'scripts/agent-customization/hooks/runtime-enforcement-hooks.test.ts'
    acceptance_criteria:
      - id: AC-014
        text: 'Runtime hook describe block is skipped when CI env is set'
    parallelizable: true
    dependencies: []
    next_slice: '02-browser-scenario'
  - slice_id: '02-browser-scenario'
    title: 'Replace docs dependency with committed fixture'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'scripts/agent-customization/browser-tests/__tests__/scenario-parser.test.ts'
      - 'testing/fixtures/webgpu-inference-smoke.html'
    acceptance_criteria:
      - id: AC-015
        text: 'scenario-parser.test.ts imports fixture path from testing/fixtures/'
      - id: AC-016
        text: 'Committed fixture contains all tokens asserted by the tests'
    parallelizable: true
    dependencies: []
    next_slice: '02-barrel-coverage'
  - slice_id: '02-barrel-coverage'
    title: 'Add barrel-import coverage for activationArrayPool, diversity, lineage'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'src/architecture/activationArrayPool.test.ts'
      - 'src/neat/neat.diversity.test.ts'
      - 'src/neat/neat.lineage.test.ts'
    acceptance_criteria:
      - id: AC-017
        text: 'New tests import every export from the three barrel files'
      - id: AC-018
        text: 'Coverage report shows 100 % statements for the three barrel files'
    parallelizable: true
    dependencies: []
    next_slice: null
```

**User instruction:** Paste this full step packet.

**Step objective:** Apply the skip/fixture/mocking decisions recorded in the
plan and add the missing barrel-file coverage tests.

**Context the agent must know:**

- Use `const isCI = Boolean(process.env.CI || process.env.GITHUB_ACTIONS);`
  and `const describeOrSkip = isCI ? describe.skip : describe;` for top-level
  guards.
- For Windows-only tests, use `const itOrSkip = process.platform === 'win32' ? it : it.skip;`
  or wrap the `it()` block with an inline guard.
- The committed fixture for the WebGPU smoke page should live in
  `testing/fixtures/webgpu-inference-smoke.html` and contain at least:
  `../../dist/neataptic.browser.iife.js`,
  `Neataptic.Network.createMLP(2, [3], 1)`, `window.webgpuSmokeResult = result;`,
  `runWebGPUSmoke`, `cpuOutput`, `gpuOutput`, `gpuDeviceBound`.

**Execution steps:**

1. In `lazy-facade.red.test.ts`:
   - Add `mkdirSync(path.dirname(tempPath), { recursive: true })` before the
     invalid-snapshot write.
   - Guard the two Node-exec-dir `npx.cmd` fallback tests with
     `process.platform === 'win32'`.
   - Add a new test that creates a temporary `npx.exe`, mocks
     `process.platform` to `'win32'`, puts the temp dir on `PATH`, and asserts
     `resolveSpawnCommand('npx')` returns `{ file: 'npx', args: [], shell: false }`.
2. In `repo-cortex-mcp.red.test.ts` and `repo-cortex-mcp.test.ts`:
   - Replace top-level `describe(...)` with
     `const describeOrSkip = isCI ? describe.skip : describe;`
     `describeOrSkip('repo cortex ...', () => { ... });`.
3. In `runtime-enforcement-hooks.test.ts`:
   - Apply the same CI guard so the SQLite index refresh never runs on CI.
4. In `scenario-parser.test.ts`:
   - Change `SCENARIO_HTML_PATH` to point to
     `testing/fixtures/webgpu-inference-smoke.html`.
   - Create the fixture with the required tokens.
5. Create new barrel coverage tests:
   - `src/architecture/activationArrayPool.test.ts`
   - `src/neat/neat.diversity.test.ts`
   - `src/neat/neat.lineage.test.ts`
     Each test imports the named exports and asserts they are defined.

**Stop conditions:**

- Done when all five slices are implemented and each targeted Jest command
  passes.
- Route back to a new slice-fix if any targeted command fails.

**Required validation:**

```bash
npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=lazy-facade.red.test.ts
npx jest --config=jest.config.mjs --no-cache --selectProjects mcp-semantic-scripts --testPathPattern=repo-cortex-mcp
npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=runtime-enforcement-hooks.test.ts
npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=scenario-parser.test.ts
npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern="activationArrayPool|neat\.diversity|neat\.lineage"
```

**Plan update requirement:** Mark each slice `[DONE]` as it passes, attach
validation evidence, and set Step 03 `[WIP]` when the step is green.

**Traceability:**

| AC id  | Criterion                                 | Files changed                                                                   | Validation command                                                                                |
| ------ | ----------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| AC-003 | lazy-facade self-check passes on CI       | `lazy-facade.red.test.ts`                                                       | `npx jest ... --testPathPattern=lazy-facade.red.test.ts`                                          |
| AC-004 | Windows-only tests skipped on non-Windows | `lazy-facade.red.test.ts`                                                       | `npx jest ... --testPathPattern=lazy-facade.red.test.ts`                                          |
| AC-005 | .exe branch covered, 100 % statements     | `lazy-facade.red.test.ts`                                                       | `npx jest ... --coverage --testPathPattern=lazy-facade.red.test.ts`                               |
| AC-006 | repo-cortex skipped on CI                 | `repo-cortex-mcp.red.test.ts`, `repo-cortex-mcp.test.ts`                        | `CI=true npx jest ... --testPathPattern=repo-cortex-mcp`                                          |
| AC-007 | runtime hooks skipped on CI               | `runtime-enforcement-hooks.test.ts`                                             | `CI=true npx jest ... --testPathPattern=runtime-enforcement-hooks.test.ts`                        |
| AC-008 | scenario-parser uses fixture              | `scenario-parser.test.ts`, `testing/fixtures/webgpu-inference-smoke.html`       | `npx jest ... --testPathPattern=scenario-parser.test.ts`                                          |
| AC-009 | barrel files 100 % statements             | `activationArrayPool.test.ts`, `neat.diversity.test.ts`, `neat.lineage.test.ts` | `npx jest ... --coverage --testPathPattern="activationArrayPool\|neat\.diversity\|neat\.lineage"` |

**Validation evidence (Step 02):**

- `npx tsc --noEmit -p tsconfig.json` → exit 0, no errors.
- `npm run lint` → exit 0, no lint errors on `src/`, `testing/`, `benchmarks/`, `examples/`.
- `npx prettier --check <changed-files>` → all matched files use Prettier code style.
- `npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=lazy-facade.red.test.ts` → 60/60 passed.
- `CI=true npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=lazy-facade.red.test.ts` → 60/60 passed (tmp dir created before write, Windows tests skipped, .exe branch test mocked win32).
- `CI=true npx jest --config=jest.config.mjs --no-cache --coverage --selectProjects agent-customization-scripts --testPathPattern=lazy-facade.red.test.ts` → 60/60 passed; `lazy-facade-core.mjs` statements/branches/functions/lines = 100/100/100/100.
- `CI=true npx jest --config=jest.config.mjs --no-cache --selectProjects mcp-semantic-scripts --testPathPattern=repo-cortex-mcp` → 2 suites skipped cleanly.
- `CI=true npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=runtime-enforcement-hooks.test.ts` → 1 suite skipped cleanly.
- `npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=scenario-parser.test.ts` → 3/3 passed using `testing/fixtures/webgpu-inference-smoke.html`.
- `npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns='activationArrayPool|neat.diversity|neat.lineage'` → 57/57 passed; `activationArrayPool.ts`, `neat.diversity.ts`, and `neat.lineage.ts` statements/branches/functions/lines = 100/100/100/100.
- `npm run agents:routing-table:ci` → pass (`matchesExpectedBody: true`, hash `a3f731a8...`).

```yaml
PlanUpdate:
  changed_files:
    - scripts/agent-customization/mcp/__tests__/lazy-facade.red.test.ts
    - scripts/mcp-semantic/__tests__/repo-cortex-mcp.red.test.ts
    - scripts/mcp-semantic/repo-cortex-mcp.test.ts
    - scripts/agent-customization/hooks/runtime-enforcement-hooks.test.ts
    - scripts/agent-customization/browser-tests/__tests__/scenario-parser.test.ts
    - testing/fixtures/webgpu-inference-smoke.html
    - src/architecture/activationArrayPool.test.ts
    - src/neat/neat.diversity.test.ts
    - src/neat/neat.lineage.test.ts
    - plans/CI_Failure_Hardening.plans.md
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check <changed-files>'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=lazy-facade.red.test.ts'
    - 'CI=true npx jest --config=jest.config.mjs --no-cache --selectProjects mcp-semantic-scripts --testPathPattern=repo-cortex-mcp'
    - 'CI=true npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=runtime-enforcement-hooks.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=scenario-parser.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns="activationArrayPool|neat.diversity|neat.lineage"'
  rollback:
    - 'git checkout -- scripts/agent-customization/mcp/__tests__/lazy-facade.red.test.ts scripts/mcp-semantic/__tests__/repo-cortex-mcp.red.test.ts scripts/mcp-semantic/repo-cortex-mcp.test.ts scripts/agent-customization/hooks/runtime-enforcement-hooks.test.ts scripts/agent-customization/browser-tests/__tests__/scenario-parser.test.ts testing/fixtures/webgpu-inference-smoke.html src/architecture/activationArrayPool.test.ts src/neat/neat.diversity.test.ts src/neat/neat.lineage.test.ts'
  next: 'Run 05-green-testing on Step 03 focused commands and attach coverage-guard evidence'
```

---

#### Step 02b: Add native-ESM Jest project for agent-customization .mjs coverage [WIP]

```yaml
phase: 1
step: '2b'
title: 'Add native-ESM Jest project for agent-customization .mjs coverage'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/CI_Failure_Hardening.plans.md'
copy_paste: true
next_step: 'Step 03 — Validate with focused Jest runs and merged coverage gate'
skills:
  - 'implementation-standards'
  - 'green-validation-gates'
validation:
  - 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=code-coverage.gate.test.mjs'
  - 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=merge-coverage-summaries.test.mjs'
  - 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=neataptic-gate-mcp.test.mjs'
  - 'node scripts/agent-customization/gates/merge-coverage-summaries.mjs'
  - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/code-coverage.gate.mjs,scripts/agent-customization/gates/merge-coverage-summaries.mjs,scripts/agent-customization/mcp/neataptic-gate-mcp.mjs'
acceptance_criteria:
  - id: AC-02b-001
    text: 'jest.config.mjs contains a project named agent-customization-mjs matching the required testMatch, transform, and V8 coverage settings'
    validation: 'node scripts/agent-customization/gates/step-packet.gate.mjs --json'
  - id: AC-02b-002
    text: 'Native-ESM tests exist for code-coverage.gate.mjs and merge-coverage-summaries.mjs under scripts/agent-customization/gates/'
    validation: 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern="code-coverage.gate.test.mjs|merge-coverage-summaries.test.mjs"'
  - id: AC-02b-003
    text: 'Native-ESM test exists for neataptic-gate-mcp.mjs under scripts/agent-customization/mcp/'
    validation: 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=neataptic-gate-mcp.test.mjs'
  - id: AC-02b-004
    text: 'The agent-customization-mjs project reports non-zero statement coverage for the target .mjs files'
    validation: 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --coverage --selectProjects agent-customization-mjs'
  - id: AC-02b-005
    text: 'merge-coverage-summaries keeps the higher per-file entry and the code-coverage gate passes for the three target .mjs files'
    validation: 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/code-coverage.gate.mjs,scripts/agent-customization/gates/merge-coverage-summaries.mjs,scripts/agent-customization/mcp/neataptic-gate-mcp.mjs'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: '02b-config'
    title: 'Add agent-customization-mjs project in jest.config.mjs'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'jest.config.mjs'
    acceptance_criteria:
      - id: AC-02b-006
        text: 'New project displayName is agent-customization-mjs with testMatch scripts/agent-customization/**/*.test.mjs'
      - id: AC-02b-007
        text: 'Project uses transform: {}, extensionsToTreatAsEsm: [.mjs], coverageProvider: v8, and coverageDirectory coverage/project-agent-customization-mjs'
      - id: AC-02b-008
        text: 'collectCoverageFrom lists the three target .mjs gate scripts'
    parallelizable: false
    dependencies: []
    next_slice: '02b-gate-tests'
  - slice_id: '02b-gate-tests'
    title: 'Write native-ESM coverage tests for gate .mjs scripts'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'scripts/agent-customization/gates/code-coverage.gate.test.mjs'
      - 'scripts/agent-customization/gates/merge-coverage-summaries.test.mjs'
    acceptance_criteria:
      - id: AC-02b-009
        text: 'code-coverage.gate.test.mjs exercises runCodeCoverageGate, main, parseGateArgs, splitPathList, deriveChangedSourceFiles, filterTargetFiles, isSourceFile, and buildFixHint'
      - id: AC-02b-010
        text: 'merge-coverage-summaries.test.mjs exercises mergeCoverageSummaries, isBetterCoverage, computeTotal, parseCliOptions, main, printMergeResult, and printMergeError'
    parallelizable: false
    dependencies:
      - '02b-config'
    next_slice: '02b-mcp-test'
  - slice_id: '02b-mcp-test'
    title: 'Write native-ESM coverage test for neataptic-gate-mcp.mjs'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'scripts/agent-customization/mcp/neataptic-gate-mcp.test.mjs'
    acceptance_criteria:
      - id: AC-02b-011
        text: 'neataptic-gate-mcp.test.mjs exercises createGateTools and runGateSelfCheck exports with a mock server'
    parallelizable: false
    dependencies:
      - '02b-gate-tests'
    next_slice: '02b-green'
  - slice_id: '02b-green'
    title: 'Validate merged coverage overrides inaccurate agent-customization-scripts summary'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - 'coverage/project-agent-customization-mjs/coverage-summary.json'
      - 'coverage/coverage-summary.json'
    acceptance_criteria:
      - id: AC-02b-012
        text: 'Running the agent-customization-mjs project produces accurate non-zero coverage for the three target .mjs files'
      - id: AC-02b-013
        text: 'merge-coverage-summaries output shows the target .mjs files came from project-agent-customization-mjs'
      - id: AC-02b-014
        text: 'code-coverage gate passes for the three target .mjs files using the merged summary'
    parallelizable: false
    dependencies:
      - '02b-mcp-test'
    next_slice: null
```

**User instruction:** Paste this full step packet.

**Step objective:** Add a dedicated native-ESM Jest project that accurately
measures coverage for the changed `scripts/agent-customization/` `.mjs` gate
scripts, because the existing `agent-customization-scripts` project's
`mjs-cjs-transformer.cjs` under `ts-jest` reports 0 % statement coverage for
those files even though the `text` reporter shows high coverage.

**Context the agent must know:**

- The existing `mcp-semantic-mjs` project already uses native ESM with
  `transform: {}` and `coverageProvider: 'v8'` to accurately cover
  `scripts/mcp-semantic/tools/submit-feedback.mjs`.
- `submit-feedback.mjs` is intentionally left in `mcp-semantic-mjs`; this step
  only adds coverage for the new/changed `.mjs` files under
  `scripts/agent-customization/`.
- `merge-coverage-summaries.mjs` keeps the most favourable per-file entry
  across projects, so the accurate `agent-customization-mjs` summary will
  override the inaccurate `agent-customization-scripts` summary for the same
  files.

**Execution steps:**

1. In `jest.config.mjs`, add a new project after `agent-customization-scripts`:
   - `displayName: 'agent-customization-mjs'`
   - `testMatch: ['**/scripts/agent-customization/**/*.test.mjs']`
   - `testEnvironment: 'node'`
   - `transform: {}`
   - `extensionsToTreatAsEsm: ['.mjs']`
   - `coverageProvider: 'v8'`
   - `coverageDirectory: 'coverage/project-agent-customization-mjs'`
   - `coverageReporters: ['lcov', 'text', 'html', 'json', 'json-summary']`
   - `collectCoverageFrom: ['scripts/agent-customization/gates/code-coverage.gate.mjs', 'scripts/agent-customization/gates/merge-coverage-summaries.mjs', 'scripts/agent-customization/mcp/neataptic-gate-mcp.mjs']`
   - `testTimeout: 300000`
2. Create `scripts/agent-customization/gates/code-coverage.gate.test.mjs` that
   dynamically imports the gate module and exercises every exported function
   and branch path (missing summary, missing baseline, no target files,
   missing file, under-covered file, CLI JSON output, etc.).
3. Create `scripts/agent-customization/gates/merge-coverage-summaries.test.mjs`
   that exercises `mergeCoverageSummaries`, `isBetterCoverage`, `computeTotal`,
   `parseCliOptions`, `main`, `printMergeResult`, and `printMergeError`.
4. Create `scripts/agent-customization/mcp/neataptic-gate-mcp.test.mjs` that
   exercises `createGateTools()` and `runGateSelfCheck({ server })` against a
   minimal mock MCP server.
5. Run the focused `agent-customization-mjs` project commands under
   `NODE_OPTIONS=--experimental-vm-modules`.
6. Run `merge-coverage-summaries.mjs` and confirm the target `.mjs` files are
   merged from `coverage/project-agent-customization-mjs/coverage-summary.json`.
7. Run `code-coverage.gate.mjs` against the three target `.mjs` files and
   confirm `pass: true` with 100 % metrics.

**Stop conditions:**

- Done when all four slices pass and the merged `code-coverage` gate is green
  for the three target `.mjs` files.
- Route back to a new `slice-fix` if any focused command fails.
- Escalate to `00.cross-tier-helper` if adding the project causes an existing
  project to fail or if `NODE_OPTIONS=--experimental-vm-modules` breaks the
  `ts-jest` projects.

**Required validation:**

```bash
NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=code-coverage.gate.test.mjs
NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=merge-coverage-summaries.test.mjs
NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=neataptic-gate-mcp.test.mjs
NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --coverage --selectProjects agent-customization-mjs
node scripts/agent-customization/gates/merge-coverage-summaries.mjs
node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/code-coverage.gate.mjs,scripts/agent-customization/gates/merge-coverage-summaries.mjs,scripts/agent-customization/mcp/neataptic-gate-mcp.mjs
```

**Plan update requirement:** Mark each slice `[DONE]` as it passes, attach
validation evidence, and set Step 03 `[WIP]` when this step is green.

**Traceability:**

| AC id      | Criterion                                                       | Files changed                                                                                                                          | Validation command                                                                                                                                                                                                                                                    |
| ---------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AC-02b-001 | jest.config.mjs project exists with required shape              | `jest.config.mjs`                                                                                                                      | `node scripts/agent-customization/gates/step-packet.gate.mjs --json`                                                                                                                                                                                                  |
| AC-02b-002 | Native-ESM tests exist for gate .mjs scripts                    | `scripts/agent-customization/gates/code-coverage.gate.test.mjs`, `scripts/agent-customization/gates/merge-coverage-summaries.test.mjs` | `NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern="code-coverage.gate.test.mjs\|merge-coverage-summaries.test.mjs"`                                                     |
| AC-02b-003 | Native-ESM test exists for gate MCP                             | `scripts/agent-customization/mcp/neataptic-gate-mcp.test.mjs`                                                                          | `NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=neataptic-gate-mcp.test.mjs`                                                                                          |
| AC-02b-004 | Non-zero statement coverage reported by agent-customization-mjs | `coverage/project-agent-customization-mjs/coverage-summary.json`                                                                       | `NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --coverage --selectProjects agent-customization-mjs`                                                                                                                             |
| AC-02b-005 | Merged summary keeps higher entry and code-coverage gate passes | `coverage/coverage-summary.json`                                                                                                       | `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/code-coverage.gate.mjs,scripts/agent-customization/gates/merge-coverage-summaries.mjs,scripts/agent-customization/mcp/neataptic-gate-mcp.mjs` |

**Validation evidence (Step 02b):**

- `npx tsc --noEmit -p tsconfig.json` → exit 0, no errors.
- `npm run lint` → exit 0 (script files are outside the lint path, as configured).
- `npx prettier --check jest.config.mjs scripts/agent-customization/gates/code-coverage.gate.mjs scripts/agent-customization/gates/code-coverage.gate.test.mjs scripts/agent-customization/gates/merge-coverage-summaries.test.mjs scripts/agent-customization/mcp/neataptic-gate-mcp.test.mjs` → all matched files use Prettier code style.
- `NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --runInBand --testPathPatterns="code-coverage.gate.test.mjs"` → 33/33 passed.
- `NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --runInBand --testPathPatterns="merge-coverage-summaries.test.mjs"` → 21/21 passed.
- `NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --runInBand --testPathPatterns="neataptic-gate-mcp.test.mjs"` → 21/21 passed.
- `NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --runInBand --coverage --coverageDirectory=coverage/project-agent-customization-mjs --testPathPatterns="code-coverage.gate.test.mjs|merge-coverage-summaries.test.mjs|neataptic-gate-mcp.test.mjs"` → 75/75 passed; `code-coverage.gate.mjs`, `merge-coverage-summaries.mjs`, and `neataptic-gate-mcp.mjs` statements/branches/functions/lines = 100/100/100/100.
- `node scripts/agent-customization/gates/merge-coverage-summaries.mjs` → merged `coverage/project-agent-customization-mjs/coverage-summary.json` and `coverage/project-agent-customization-scripts/coverage-summary.json` into `coverage/coverage-summary.json`.
- `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/code-coverage.gate.mjs,scripts/agent-customization/gates/merge-coverage-summaries.mjs,scripts/agent-customization/mcp/neataptic-gate-mcp.mjs` → `pass: true` with all four metrics at 100% for the three target files.
- `node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --self-check --json` → `ok: true`.

**Tooling notes:** Jest 30.2.0 rejects `extensionsToTreatAsEsm: ['.mjs']` (`.mjs` is always treated as an ECMAScript Module) and emits a validation warning for project-level `coverageProvider: 'v8'`, but the native-ESM project still produces accurate V8-style coverage. In this environment the project-level `coverageDirectory` is written only when the `--coverageDirectory` CLI flag is supplied, so the green run uses `--coverageDirectory=coverage/project-agent-customization-mjs` to place the per-project summary where `merge-coverage-summaries.mjs` expects it. The step packet lists `--testPathPattern` (singular); the local Jest CLI requires the plural `--testPathPatterns`.

```yaml
PlanUpdate:
  changed_files:
    - jest.config.mjs
    - scripts/agent-customization/gates/code-coverage.gate.mjs
    - scripts/agent-customization/gates/code-coverage.gate.test.mjs
    - scripts/agent-customization/gates/merge-coverage-summaries.test.mjs
    - scripts/agent-customization/mcp/neataptic-gate-mcp.test.mjs
    - plans/CI_Failure_Hardening.plans.md
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check jest.config.mjs scripts/agent-customization/gates/code-coverage.gate.mjs scripts/agent-customization/gates/code-coverage.gate.test.mjs scripts/agent-customization/gates/merge-coverage-summaries.test.mjs scripts/agent-customization/mcp/neataptic-gate-mcp.test.mjs'
  tests_for_green:
    - 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --runInBand --testPathPatterns="code-coverage.gate.test.mjs|merge-coverage-summaries.test.mjs|neataptic-gate-mcp.test.mjs"'
    - 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --runInBand --coverage --coverageDirectory=coverage/project-agent-customization-mjs --testPathPatterns="code-coverage.gate.test.mjs|merge-coverage-summaries.test.mjs|neataptic-gate-mcp.test.mjs"'
    - 'node scripts/agent-customization/gates/merge-coverage-summaries.mjs'
    - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/code-coverage.gate.mjs,scripts/agent-customization/gates/merge-coverage-summaries.mjs,scripts/agent-customization/mcp/neataptic-gate-mcp.mjs'
    - 'node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --self-check --json'
  rollback:
    - 'git checkout -- jest.config.mjs scripts/agent-customization/gates/code-coverage.gate.mjs scripts/agent-customization/gates/code-coverage.gate.test.mjs scripts/agent-customization/gates/merge-coverage-summaries.test.mjs scripts/agent-customization/mcp/neataptic-gate-mcp.test.mjs plans/CI_Failure_Hardening.plans.md'
  next: 'Run 05-green-testing on Step 03 focused commands and attach coverage-guard evidence'
```

---

#### Step 03: Validate with focused Jest runs and merged coverage gate [DONE]

```yaml
phase: 1
step: 3
title: 'Validate with focused Jest runs and merged coverage gate'
goal: 'green-testing'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/CI_Failure_Hardening.plans.md'
copy_paste: true
next_step: 'Step 04 — Close the phase and compress to logs'
status: '[DONE]'
skills:
  - 'green-validation-gates'
  - 'code-quality-auditor'
validation:
  - 'npm run lint'
  - 'npm run agents:routing-table:ci'
  - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json'
  - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=src/architecture/activationArrayPool.ts,src/neat/neat.diversity.ts,src/neat/neat.lineage.ts'
  - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/code-coverage.gate.mjs,scripts/agent-customization/gates/merge-coverage-summaries.mjs,scripts/agent-customization/mcp/neataptic-gate-mcp.mjs'
  - 'node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --self-check --json'
  - 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=code-coverage.gate.test.mjs'
  - 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=merge-coverage-summaries.test.mjs'
  - 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=neataptic-gate-mcp.test.mjs'
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=lazy-facade.red.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects mcp-semantic-scripts --testPathPattern=repo-cortex-mcp'
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=runtime-enforcement-hooks.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=scenario-parser.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern="activationArrayPool|neat\\.diversity|neat\\.lineage"'
acceptance_criteria:
  - id: AC-019
    text: 'Lint passes with zero errors'
    validation: 'npm run lint'
  - id: AC-020
    text: 'Routing table freshness gate passes'
    validation: 'npm run agents:routing-table:ci'
  - id: AC-021b
    text: 'The new code-coverage gate exists, is registered in the gate MCP, and passes against the three barrel files'
    validation: 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=src/architecture/activationArrayPool.ts,src/neat/neat.diversity.ts,src/neat/neat.lineage.ts'
  - id: AC-021c
    text: 'The code-coverage gate passes for the three new agent-customization .mjs files using the merged summary'
    validation: 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/code-coverage.gate.mjs,scripts/agent-customization/gates/merge-coverage-summaries.mjs,scripts/agent-customization/mcp/neataptic-gate-mcp.mjs'
  - id: AC-021d
    text: 'All agent-customization-mjs focused tests pass under native ESM'
    validation: 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern="code-coverage.gate.test.mjs|merge-coverage-summaries.test.mjs|neataptic-gate-mcp.test.mjs"'
  - id: AC-021
    text: 'All touched test files pass focused Jest runs'
    validation: 'npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts,mcp-semantic-scripts --testPathPattern="lazy-facade|repo-cortex-mcp|runtime-enforcement-hooks|scenario-parser|activationArrayPool|neat\\.diversity|neat\\.lineage"'
constitution_check:
  - 'principle-4-small-slices'
```

**User instruction:** Paste this full step packet.

**Step objective:** Re-validate the permanent `code-coverage` Tier-1 gate and
all focused Jest runs after Step 02b adds the native-ESM
`agent-customization-mjs` project. Confirm lint, routing-table freshness, the
code-coverage gate (for both barrel files and the new `.mjs` gate scripts), the
gate MCP self-check, and every focused project run remain green.

**Required validation:**

```bash
npm run lint
npm run agents:routing-table:ci
node scripts/agent-customization/gates/code-coverage.gate.mjs --json
node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=src/architecture/activationArrayPool.ts,src/neat/neat.diversity.ts,src/neat/neat.lineage.ts
node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/code-coverage.gate.mjs,scripts/agent-customization/gates/merge-coverage-summaries.mjs,scripts/agent-customization/mcp/neataptic-gate-mcp.mjs
node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --self-check --json
NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=code-coverage.gate.test.mjs
NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=merge-coverage-summaries.test.mjs
NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --testPathPattern=neataptic-gate-mcp.test.mjs
npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=lazy-facade.red.test.ts
npx jest --config=jest.config.mjs --no-cache --selectProjects mcp-semantic-scripts --testPathPattern=repo-cortex-mcp
npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=runtime-enforcement-hooks.test.ts
npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --testPathPattern=scenario-parser.test.ts
npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern="activationArrayPool|neat\.diversity|neat\.lineage"
```

**Plan update requirement:** Record all green validation evidence, then mark
Step 03 `[DONE]` and Step 04 `[WIP]`.

**Evidence:**

- `code-coverage` gate created at
  `scripts/agent-customization/gates/code-coverage.gate.mjs`.
- Gate registered in `scripts/agent-customization/mcp/neataptic-gate-mcp.mjs`
  `TIER_1_GATES` with owner `code-coverage.gate.mjs`.
- `green-validation-gates` skill catalog updated to list `code-coverage`.
- `05-green-testing` agent Constraints updated to mandate the `code-coverage`
  gate for `src/` and `scripts/agent-customization/` changes.
- Sample run against the three barrel files returns `pass: true` with all four
  metrics at 100 % (see gate output below).

```json
{
  "pass": true,
  "evidence": {
    "coverageSummaryPath": "coverage/coverage-summary.json",
    "targetFiles": [
      "src/architecture/activationArrayPool.ts",
      "src/neat/neat.diversity.ts",
      "src/neat/neat.lineage.ts"
    ],
    "fileReports": [
      {
        "file": "src/architecture/activationArrayPool.ts",
        "found": true,
        "metrics": {
          "lines": 100,
          "statements": 100,
          "functions": 100,
          "branches": 100
        },
        "allCovered": true
      },
      {
        "file": "src/neat/neat.diversity.ts",
        "found": true,
        "metrics": {
          "lines": 100,
          "statements": 100,
          "functions": 100,
          "branches": 100
        },
        "allCovered": true
      },
      {
        "file": "src/neat/neat.lineage.ts",
        "found": true,
        "metrics": {
          "lines": 100,
          "statements": 100,
          "functions": 100,
          "branches": 100
        },
        "allCovered": true
      }
    ],
    "missingFiles": [],
    "failedFiles": []
  },
  "fixHint": null,
  "owner": "code-coverage"
}
```

**Validation evidence (Step 03):**

- `npx tsc --noEmit -p tsconfig.json` → exit 0, no errors.
- `npm run lint` → exit 0, no lint errors on `src/`, `testing/`, `benchmarks/`, `examples/`.
- `npx prettier --check scripts/agent-customization/gates/merge-coverage-summaries.mjs scripts/agent-customization/gates/merge-coverage-summaries.test.mjs scripts/agent-customization/gates/code-coverage.gate.mjs jest.config.mjs` → all matched files use Prettier code style.
- Per-project coverage recipe (run to populate `coverage/project-*/coverage-summary.json`):
  - `npx jest --config=jest.config.mjs --no-cache --selectProjects default --coverage --coverageDirectory=coverage/project-default --testPathPatterns="activationArrayPool|neat\.diversity|neat\.lineage|network\.gpu"` → exit 0; produced `coverage/project-default/coverage-summary.json`.
  - `npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --coverage --coverageDirectory=coverage/project-agent-customization-scripts --testPathPatterns="scripts/agent-customization/gates/(code-coverage\.gate\.test\.ts|merge-coverage-summaries\.gate\.test\.ts)"` → exit 0; produced `coverage/project-agent-customization-scripts/coverage-summary.json`.
  - `$env:NODE_OPTIONS='--experimental-vm-modules'; npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --coverage --coverageDirectory=coverage/project-agent-customization-mjs --testPathPatterns="code-coverage\.gate\.test\.mjs|merge-coverage-summaries\.test\.mjs|neataptic-gate-mcp\.test\.mjs"` → 79/79 passed; `code-coverage.gate.mjs`, `merge-coverage-summaries.mjs`, and `neataptic-gate-mcp.mjs` statements/branches/functions/lines = 100/100/100/100.
  - `$env:NODE_OPTIONS='--experimental-vm-modules'; npx jest --config=jest.config.mjs --no-cache --selectProjects mcp-semantic-mjs --coverage --coverageDirectory=coverage/project-mcp-semantic-mjs --testPathPatterns="submit-feedback"` → 16/16 passed; `submit-feedback.mjs` statements/branches/functions/lines = 100/100/100/100.
  - `npx jest --config=jest.config.mjs --no-cache --selectProjects mcp-semantic-scripts --coverage --coverageDirectory=coverage/project-mcp-semantic-scripts --testPathPatterns="submit-feedback"` → exit 0; produced `coverage/project-mcp-semantic-scripts/coverage-summary.json`.
- `node scripts/agent-customization/gates/merge-coverage-summaries.mjs --input coverage --output coverage/coverage-summary.json --baseline=coverage/coverage-baseline.json` → merged 5 per-project summaries into `coverage/coverage-summary.json` and generated `coverage/coverage-baseline.json` from `git status --porcelain`; 9 changed source files recorded, missing files default to 0 %.
- Removed the temporary `tmp/generate-coverage-baseline.mjs`; baseline generation now lives in `merge-coverage-summaries.mjs`.
- `node scripts/agent-customization/gates/code-coverage.gate.mjs --json` → `pass: true` for git-derived changed files; 9 target files, 0 missing, 0 failed.
- `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=src/architecture/activationArrayPool.ts,src/neat/neat.diversity.ts,src/neat/neat.lineage.ts` → `pass: true`; all three barrel files 100/100/100/100.
- `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/code-coverage.gate.mjs,scripts/agent-customization/gates/merge-coverage-summaries.mjs,scripts/agent-customization/mcp/neataptic-gate-mcp.mjs` → `pass: true`; all three gate `.mjs` files 100/100/100/100.
- `node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --self-check --json` → `ok: true`, 0 errors, 0 warnings.
- `npm run agents:routing-table:ci` → pass (`matchesExpectedBody: true`, hash `78ddb7f77621d40ef60ba4ec70e07bb1c0f43ed9c296e52ffb21c1200964e108`).

```yaml
PlanUpdate:
  changed_files:
    - scripts/agent-customization/gates/merge-coverage-summaries.mjs
    - scripts/agent-customization/gates/merge-coverage-summaries.test.mjs
    - scripts/agent-customization/gates/code-coverage.gate.test.ts
    - scripts/agent-customization/gates/code-coverage.gate.test.mjs
    - plans/CI_Failure_Hardening.plans.md
    - coverage/coverage-baseline.json
    - coverage/coverage-summary.json
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
    - 'npx prettier --check scripts/agent-customization/gates/merge-coverage-summaries.mjs scripts/agent-customization/gates/merge-coverage-summaries.test.mjs scripts/agent-customization/gates/code-coverage.gate.mjs scripts/agent-customization/gates/code-coverage.gate.test.mjs scripts/agent-customization/gates/code-coverage.gate.test.ts jest.config.mjs'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --coverage --coverageDirectory=coverage/project-default --testPathPatterns="activationArrayPool|neat\.diversity|neat\.lineage|network\.gpu"'
    - 'npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-scripts --coverage --coverageDirectory=coverage/project-agent-customization-scripts --testPathPatterns="scripts/agent-customization/gates/(code-coverage\.gate\.test\.ts|merge-coverage-summaries\.gate\.test\.ts)"'
    - 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects agent-customization-mjs --coverage --coverageDirectory=coverage/project-agent-customization-mjs --testPathPatterns="code-coverage\.gate\.test\.mjs|merge-coverage-summaries\.test\.mjs|neataptic-gate-mcp\.test\.mjs"'
    - 'NODE_OPTIONS=--experimental-vm-modules npx jest --config=jest.config.mjs --no-cache --selectProjects mcp-semantic-mjs --coverage --coverageDirectory=coverage/project-mcp-semantic-mjs --testPathPatterns="submit-feedback"'
    - 'npx jest --config=jest.config.mjs --no-cache --selectProjects mcp-semantic-scripts --coverage --coverageDirectory=coverage/project-mcp-semantic-scripts --testPathPatterns="submit-feedback"'
    - 'node scripts/agent-customization/gates/merge-coverage-summaries.mjs --input coverage --output coverage/coverage-summary.json --baseline=coverage/coverage-baseline.json'
    - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json'
    - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=src/architecture/activationArrayPool.ts,src/neat/neat.diversity.ts,src/neat/neat.lineage.ts'
    - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/code-coverage.gate.mjs,scripts/agent-customization/gates/merge-coverage-summaries.mjs,scripts/agent-customization/mcp/neataptic-gate-mcp.mjs'
    - 'npm run agents:routing-table:ci'
  rollback:
    - 'git checkout -- scripts/agent-customization/gates/merge-coverage-summaries.mjs scripts/agent-customization/gates/merge-coverage-summaries.test.mjs scripts/agent-customization/gates/code-coverage.gate.test.ts scripts/agent-customization/gates/code-coverage.gate.test.mjs plans/CI_Failure_Hardening.plans.md'
    - 'rm coverage/coverage-baseline.json coverage/coverage-summary.json'
  next: 'Run 05-green-testing to confirm the Step 03 focused commands, then advance to Step 04 (phase compression and archive).'
```

**05-green-testing final confirmation (this run):**

- `npx tsc --noEmit -p tsconfig.json` → exit 0
- `npx tsc --noEmit -p tsconfig.test.json` → exit 0
- `npm run lint` → exit 0
- `npx prettier --check` over all touched gate files → exit 0
- `npx jest ... --selectProjects default` → 17 suites / 257 tests passed
- `npx jest ... --selectProjects agent-customization-scripts` → 51/51 passed
- `npx jest ... --selectProjects agent-customization-mjs --runInBand` → 91/91 passed; `code-coverage.gate.mjs`, `merge-coverage-summaries.mjs`, `neataptic-gate-mcp.mjs` all 100/100/100/100
- `npx jest ... --selectProjects mcp-semantic-mjs` → 16/16 passed
- `npx jest ... --selectProjects mcp-semantic-scripts` → exit 0
- `merge-coverage-summaries.mjs --baseline=...` → merged 5 summaries; baseline contains 9 changed files
- `code-coverage.gate.mjs --json` → pass: true; 9 target files, 0 missing, 0 failed
- `code-coverage.gate.mjs --json --changed-files=src/architecture/activationArrayPool.ts,src/neat/neat.diversity.ts,src/neat/neat.lineage.ts` → pass: true; all 100/100/100/100
- `code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/code-coverage.gate.mjs,scripts/agent-customization/gates/merge-coverage-summaries.mjs,scripts/agent-customization/mcp/neataptic-gate-mcp.mjs` → pass: true; all 100/100/100/100
- `npm run agents:routing-table:ci` → pass: true
- `node scripts/agent-customization/gates/plan-sync.gate.mjs --json` → pass: true
- `node scripts/agent-customization/gates/step-packet.gate.mjs --plan plans/CI_Failure_Hardening.plans.md --json` → pass: true
- `node scripts/agent-customization/gates/agent-graph.gate.mjs --json` → pass: true
- Independent `coverage-guard` specialist re-run → all 6 changed source files 100/100/100/100; code-coverage gate pass: true
- Independent `code-quality-auditor` specialist re-run → build/lint/prettier/routing/plan-sync/step-packet all green

---

#### Step 04: Close the phase and compress to logs [WIP]

```yaml
phase: 1
step: 4
title: 'Close the phase and compress to logs'
goal: 'logging'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/CI_Failure_Hardening.plans.md'
copy_paste: true
next_step: 'Phase 1 complete — archived to plans/completed/'
status: '[WIP]'
skills:
  - 'tracker-handoff'
validation:
  - 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
  - 'node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json'
  - 'node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/CI_Failure_Hardening.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/CI_Failure_Hardening.plans.md'
acceptance_criteria:
  - id: AC-022
    text: 'Phase compression gate confirms completed phase is concise'
    validation: 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
  - id: AC-023
    text: 'Log completion marker exists for the workstream'
    validation: 'node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json'
  - id: AC-024
    text: 'No stale top-level [WIP] marker remains after archive'
    validation: 'node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json'
  - id: AC-024b
    text: 'Plan registration and phase packets remain valid after closure updates'
    validation: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/CI_Failure_Hardening.plans.md'
constitution_check:
  - 'principle-4-small-slices'
```

**User instruction:** Paste this full step packet.

**Step objective:** Finalize CI Failure Hardening closure after Step 02b and
Step 03 are green: update `plans/Roadmap.md` to `[DONE]`, repoint
`.vscode/mcp.json` if needed, record the final `code-coverage` gate evidence,
and compress the phase into a log entry under `plans/completed/`.

**Required validation:**

```bash
node scripts/agent-customization/gates/phase-compression.gate.mjs --json
node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json
node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json
node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/CI_Failure_Hardening.plans.md
node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/CI_Failure_Hardening.plans.md
```

**Plan update requirement:** Move `plans/CI_Failure_Hardening.plans.md` to
`plans/completed/` after the gates pass. (No `.logs.md` file was produced for
this workstream.)

**Note:** This step was previously completed and then reopened to add the
`agent-customization-mjs` native-ESM coverage project. The closure evidence
below is stale and will be refreshed when Step 04 is run again.

**Stale closure evidence (from prior close-out pass, to be refreshed):**

- Previously moved `plans/CI_Failure_Hardening.plans.md` to `plans/completed/CI_Failure_Hardening.plans.md`.
- Previously updated `plans/README.md` to list the plan under `plans/completed/` with `[DONE]`.
- Validations run at that time:
  - `npm run lint` → PASS
  - `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json` → PASS
  - `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/completed/CI_Failure_Hardening.plans.md` → PASS
  - `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/completed/CI_Failure_Hardening.plans.md` → PASS
  - `npm run agents:routing-table:gate` → PASS
  - `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` → FAIL (pre-existing: unknown skills `browser-testing-harness` and `checkpointing-persistence` in agents not touched by this workstream)
  - `node scripts/agent-customization/validate-skill-frontmatter.mjs --json` → PASS
  - Gate MCP `agent-graph` → PASS
  - Gate MCP `cortex-index` → index freshness PASS; workflow binding depends on active plan state at runtime.

---

#### Step 05: Documentation update — skipped [PLANNED]

```yaml
phase: 1
step: 5
goal: 'documenting'
status: '[PLANNED]'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/CI_Failure_Hardening.plans.md'
copy_paste: false
next_step: null
skills:
  - 'educational-docs'
validation:
  - 'Skipped: no user-facing documentation changes required for test hardening'
acceptance_criteria:
  - id: AC-025
    text: 'No README or generated docs changes are required'
    validation: 'Manual confirmation'
constitution_check:
  - 'principle-4-small-slices'
```

**Skip rationale:** This workstream only changes tests and a test fixture.
No public API or tutorial documentation needs updating. The fixture path is
self-documenting by its location under `testing/fixtures/`.

---

## Decision record

```yaml
decision_record:
  id: 'DR-2026-07-07-01'
  context: 'CI has no SQLite, GPU, Windows binaries, or prebuilt docs/dist. Several integration tests and one browser scenario parser assume those resources exist.'
  options:
    - id: skip
      desc: 'Skip local-only tests on CI via env guard'
    - id: mock
      desc: 'Mock SQLite / GPU / Windows filesystem in every test'
    - id: rename
      desc: 'Rename tests to .local.test.ts so Jest ignores them on CI'
  chosen: skip
  rationale: 'The repo-cortex and runtime-enforcement-hooks tests are integration contracts that validate real tool behavior against the local corpus and SQLite index. Mocking them would defeat their purpose. Renaming would hide them from local runs too. A CI skip guard keeps them meaningful for local development while making CI green. The docs-dependent test is replaced with a committed fixture because its only purpose is to assert static token presence.'
  owner: '01-planning'
  rollback_plan: 'Remove the env guards or convert them to mock-based fixtures if the project later wants CI to exercise the real corpus.'
  created_at: '2026-07-07T19:36:21-04:00'
```

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Workstream: CI Failure Hardening.
Active plan: plans/CI_Failure_Hardening.plans.md.
Current boundary: Step 02b is [WIP] — add a native-ESM Jest project so changed `scripts/agent-customization/` `.mjs` gate scripts are measured accurately. Step 02 is already [DONE]; Step 03 and Step 04 are [PLANNED] pending Step 02b completion.

Already covered:
- CI failures inspected and classified (skip vs fixture).
- Step 02 slices implemented and validated.
- Step 02b planning packet authored and gate-checked (see `## Latest validation evidence`).

Next narrow task:
1. Implement slice 02b-config (add `agent-customization-mjs` Jest project in `jest.config.mjs`).
2. Implement slice 02b-gate-tests (native-ESM tests for `code-coverage.gate.mjs`, `merge-coverage-summaries.mjs`, `step-packet.gate.mjs`).
3. Implement slice 02b-mcp-test (native-ESM test for `mcp/neataptic-gate-mcp.mjs`).
4. Implement slice 02b-green (run focused new project, verify merged coverage, run lint).

Required validations per slice are listed in Step 02b.
After all slices pass, run Step 03 validations (focused `agent-customization-mjs`, merged `code-coverage` gate, lint, plan-sync).
```

## PlanUpdate

```yaml
PlanUpdate:
  phase: 1
  status: WIP
  step: '02b'
  evidence:
    - 'Reopened plans/CI_Failure_Hardening.plans.md from plans/completed/.'
    - 'Added Step 02b packet: native-ESM Jest project for agent-customization .mjs coverage.'
    - 'Updated Step 03 validation to include focused agent-customization-mjs runs and merged code-coverage gate.'
    - 'Reverted Step 04 to [PLANNED] and repointed source_of_truth to plans/.'
    - 'Updated plans/README.md and plans/Roadmap.md to list CI Failure Hardening as [WIP].'
    - 'Gate checks recorded in `## Latest validation evidence`.'
  next_agent: '04-implementing'
```
