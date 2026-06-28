# RAG/index infrastructure reorganization

**Status:** [WIP]

## Purpose

Consolidate all Repo Cortex RAG/index scripts, generated databases, model caches,
freshness proofs, hook contexts, and browser snapshots into a single top-level
`rag-index/` directory. Replace the scattered `scripts/semantic-index/` +
`data/` + root `freshness-proof-*/` layout with one coherent folder, add a
single idempotent incremental entry point (`rag-index/update-rag.mjs`), and
update all downstream consumers (`package.json`, `.gitignore`, MCP wiring,
workflow skills/agents, validation gates).

## Scope

In scope:

- Move `scripts/semantic-index/` → `rag-index/`, preserving internal subfolder
  structure (`__tests__/`, `code-quality/`, `docs-quality/`, `embed/`, `eval/`,
  `freshness-hooks/`, `rag-eval/`, `ts-chunk/`).
- Move generated/local artifacts into `rag-index/`:
  - `data/turso-replica.sqlite` → `rag-index/data/turso-replica.sqlite`
  - `data/hook-context-*.json` → `rag-index/data/hook-context/`
  - `data/mcp-session-override.json` → `rag-index/data/mcp-session-override.json`
  - `scripts/semantic-index/models/` → `rag-index/models/`
  - root `freshness-proof-*/` → `rag-index/freshness-proofs/`
- Change default snapshot output from `docs/assets/semantic-snapshot.json` to
  `rag-index/snapshots/semantic-snapshot.json`.
- Update default path constants in the moved scripts (`init-schema.mjs`,
  `embed-index.mjs`, `reranker-readiness.mjs`, `build-browser-snapshot.mjs`,
  `runtime-enforcement.mjs`).
- Update child-process spawn paths in `prewarm-dense.mjs`,
  `session-start-index.mjs`, `session-start-cortex-mcp-preflight.mjs`,
  `refresh-cortex-after-write.mjs`.
- Update imports/fix hints in `scripts/mcp-semantic/tools/cortex-db.mjs`,
  `scripts/agent-customization/gates/cortex-index.gate.mjs`, and other
  agent-customization gates/hooks.
- Update `package.json` RAG npm scripts, add `rag:update`.
- Update `jest.config.mjs` project test-match patterns.
- Update `.gitignore`, `.vscode/mcp.json`, `.mcp.json`.
- Update `.github/skills/repo-cortex-workflow/SKILL.md` and
  `.github/agents/repo-cortex-scout.agent.md` path examples.
- Create `plans/rag-update.plans.md` (this file) and register it in
  `plans/README.md` and `plans/Roadmap.md`.

Out of scope (do not move or modify):

- `src/` library API, examples, benchmarks.
- Cloud/Turso deployment topology beyond the local default path.
- Core NEAT algorithms.
- Historical completed plans under `plans/completed/` and the immutable
  `.github/ai-learning/learning-log.jsonl`.
- `data/eval-baselines/` — persistent evaluation data, not a generated runtime
  artifact; leave in `data/`.

## Current state

Investigation complete. No files have been moved or edited yet. The
`cortex-index`, `plan-sync`, and `step-packet` gates are green against the
existing layout. Subagent briefs from `boundary-mapper`,
`planning-risk-coordinator`, and `acceptance-criteria-writer` are attached as
context for the implementation pass.

Known constraints:

- `build-entity-graph.mjs` currently runs `DELETE FROM edges` / `DELETE FROM
entities` on every invocation, so the unified script will run it as a
  conditional full rebuild rather than a true incremental stage.
- The live `cortex` MCP server uses `.vscode/mcp.json` `TURSO_DATABASE_URL` to
  override the compiled-in default; the config must change in lockstep with
  `defaultDatabasePath`.
- The DB is ~530 MB and ONNX caches are ~180 MB; moves must preserve file
  integrity.

## Decision records

```yaml
decision_record:
  id: 'DR-2026-06-29-01'
  context: 'How to structure the new rag-index/ directory.'
  options:
    - id: flat
      desc: 'Keep the existing scripts/semantic-index/ internal tree intact and rename the root; add rag-index/data/, models/, freshness-proofs/, snapshots/ alongside it.'
    - id: grouped
      desc: 'Reshuffle top-level scripts into semantic chapter folders (corpus/, embed/, query/, graph/, eval/, shared/).'
  chosen: flat
  rationale: 'Minimizes relative-import churn, keeps tests discoverable under the same jest project rename, and makes the move a near-atomic rename. A future refactor can introduce semantic grouping once the consolidation is stable.'
  owner: '01-planning'
  rollback_plan: 'If downstream imports break unexpectedly, restore the original tree and retry with the grouped approach in a follow-up plan.'
  created_at: '2026-06-29T00:00:00Z'
```

```yaml
decision_record:
  id: 'DR-2026-06-29-02'
  context: 'Where to emit the browser-facing semantic snapshot.'
  options:
    - id: docs-assets
      desc: 'Keep default output at docs/assets/semantic-snapshot.json because browser consumers and existing skills reference it there.'
    - id: rag-index
      desc: 'Move default output to rag-index/snapshots/semantic-snapshot.json to consolidate all generated index artifacts under rag-index/.'
  chosen: rag-index
  rationale: 'The user explicitly asked to move snapshots into rag-index/. Browser consumers and skill examples will be updated to the new path; the snapshot remains a generated artifact, not a hand-maintained docs source.'
  owner: '01-planning'
  rollback_plan: 'If docs/browser pipeline cannot consume the new path, add a copy step from rag-index/snapshots/ to docs/assets/ and revert the default.'
  created_at: '2026-06-29T00:00:00Z'
```

```yaml
decision_record:
  id: 'DR-2026-06-29-03'
  context: 'How to handle the non-incremental entity-graph stage.'
  options:
    - id: refactor-first
      desc: 'Make build-entity-graph.mjs incremental before adding it to update-rag.mjs.'
    - id: wire-as-is
      desc: 'Run build-entity-graph.mjs in the chain and document it as a conditional full rebuild.'
  chosen: wire-as-is
  rationale: 'The user asked to chain existing operations. Adding true incrementality is a separate behavior change that should be planned, red-tested, and measured independently. The unified script will skip the graph stage when no corpus changes are detected, and document that the stage itself performs a full rebuild when it runs.'
  owner: '01-planning'
  rollback_plan: 'Revisit in a follow-up plan focused on incremental entity-graph updates.'
  created_at: '2026-06-29T00:00:00Z'
```

---

## Implementation phases

### Phase 1 — Consolidate RAG/index infrastructure into rag-index/ [WIP]

```yaml
phase: 1
title: 'Consolidate RAG/index infrastructure into rag-index/'
status: '[WIP]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/rag-update.plans.md'
copy_paste: true
next_phase: 'Archive to plans/completed/ once all steps reach [DONE]'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
  - 'implementation-standards'
  - 'phase-handoff-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/rag-update.plans.md'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/rag-update.plans.md'
acceptance_criteria:
  - 'All required reorganization work is represented as steps or explicit skips.'
  - 'Plan file passes structural validation and plan-sync registration gates.'
  - 'Decision records resolve folder layout, snapshot location, and graph-stage incrementality.'
placeholder_steps:
  - 'Step 01 — Planning the reorganization'
  - 'Step 02 — Relocate scripts and generated artifacts into rag-index/'
  - 'Step 03 — Update path constants and internal imports'
  - 'Step 04 — Create rag-index/update-rag.mjs idempotent orchestrator'
  - 'Step 05 — Update package.json, jest.config.mjs, and .gitignore'
  - 'Step 06 — Update MCP wiring, gates, hooks, skills, and agents'
  - 'Step 07 — Clean stale temporal artifacts and register the plan'
  - 'Step 08 — Green validation'
```

**Phase objective:** Move the entire RAG/index surface into `rag-index/`, update
all path constants and consumers, add the unified `update-rag.mjs` script, and
prove the stack is still green via the `cortex-index` gate and focused tests.

**Stop conditions:**

- Any path change that breaks the live `cortex` MCP server or the
  `cortex-index` gate is fixed in the same step.
- No old code, fix hints, npm scripts, or `.gitignore` entries that reference
  the old layout are left behind.
- No deferred cleanup: old locations are removed in the same step that
  introduces the new ones.

**Required validation:**

- `validate-plan-phase-packets` and `validate-plan-sync` must pass for this plan.
- `cortex-index` gate must pass after the relocation.
- `npm run rag:update -- --dry-run` must list stages without errors.
- `npx jest --config=jest.config.mjs --selectProjects rag-index-scripts --selectProjects rag-index-mjs` must pass.
- `npm run lint` must pass on touched files.

---

#### Step 01: Planning the reorganization [DONE]

```yaml
phase: 1
step: 1
title: 'Planning the reorganization'
status: '[DONE]'
goal: 'planning'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/rag-update.plans.md'
copy_paste: true
next_step: 'Step 02 — Relocate scripts and generated artifacts into rag-index/'
owner: 'Copilot'
reviewer: 'Maintainer'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
  - 'plan-sync-validation'
  - 'execute'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/rag-update.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-sync'
acceptance_criteria:
  - 'plans/rag-update.plans.md exists and conforms to the step-packet schema.'
  - 'plans/README.md and plans/Roadmap.md are updated to register this plan (see Step 07 for the actual edit; this step prepares the registration packets).'
  - 'validate-plan-sync passes with 0 errors.'
  - 'Decision records are recorded for folder layout, snapshot location, and graph-stage incrementality.'
```

**User instruction:** Execute this step from the current repo state without relying on prior chat history. Stop and escalate to 00-helping if any downstream consumer breaks or if large artifacts are not copied intact.

**Step objective:** Author the durable tracker and decide the target layout
before any file moves.

**Required validation:** Run `validate-plan-sync` and confirm the plan is
structurally valid.

---

#### Step 02: Relocate scripts and generated artifacts into rag-index/ [PLANNED]

```yaml
phase: 1
step: 2
title: 'Relocate scripts and generated artifacts into rag-index/'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/rag-update.plans.md'
copy_paste: true
next_step: 'Step 03 — Update path constants and internal imports'
owner: '04-implementing'
reviewer: 'Maintainer'
skills:
  - 'implementation-standards'
  - 'tracker-handoff'
validation:
  - 'git ls-files scripts/semantic-index/ | wc -l  # expect 0'
  - 'git ls-files rag-index/ | wc -l              # expect > 0'
  - 'node -e "import(\'./rag-index/init-schema.mjs\').then(m=>console.log(m.defaultDatabasePath))"'
acceptance_criteria:
  - 'All tracked RAG source/test files live under rag-index/.'
  - 'The corpus DB, model cache, hook contexts, freshness proofs, and snapshots live under rag-index/.'
  - 'No tracked files remain under scripts/semantic-index/ or in old artifact locations.'
slices:
  - slice_id: '02-move-scripts'
    title: 'Move scripts/semantic-index/ tree to rag-index/'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'scripts/semantic-index/**'
      - 'rag-index/**'
    acceptance_criteria:
      - 'Directory tree is moved as a whole; internal subfolders preserved.'
      - 'git ls-files scripts/semantic-index/ returns empty.'
    parallelizable: false
    dependencies: []
    next_slice: '02-move-db'
  - slice_id: '02-move-db'
    title: 'Move data/turso-replica.sqlite and hook contexts into rag-index/data/'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - 'data/turso-replica.sqlite'
      - 'data/hook-context-*.json'
      - 'data/mcp-session-override.json'
      - 'rag-index/data/'
    acceptance_criteria:
      - 'DB and hook-context files are present under rag-index/data/.'
      - 'No tracked or untracked copies remain at the old data/ locations (except data/eval-baselines/).'
    parallelizable: false
    dependencies:
      - '02-move-scripts'
    next_slice: '02-move-models'
  - slice_id: '02-move-models'
    title: 'Move ONNX model cache into rag-index/models/'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - 'scripts/semantic-index/models/**'
      - 'rag-index/models/**'
    acceptance_criteria:
      - 'Model files and reranker/ subdirectory are present under rag-index/models/.'
      - 'No model files remain under scripts/semantic-index/models/.'
    parallelizable: false
    dependencies:
      - '02-move-scripts'
    next_slice: '02-move-proofs'
  - slice_id: '02-move-proofs'
    title: 'Move root freshness-proof-* directories into rag-index/freshness-proofs/'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - 'freshness-proof-*/'
      - 'rag-index/freshness-proofs/'
    acceptance_criteria:
      - 'All freshness-proof directories are under rag-index/freshness-proofs/.'
      - 'No freshness-proof-* directories remain at repo root.'
    parallelizable: false
    dependencies:
      - '02-move-scripts'
    next_slice: '02-create-snapshots'
  - slice_id: '02-create-snapshots'
    title: 'Create rag-index/snapshots/ directory for browser snapshot output'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - 'rag-index/snapshots/'
      - 'rag-index/build-browser-snapshot.mjs'
    acceptance_criteria:
      - 'rag-index/snapshots/ directory exists and is gitignored.'
      - 'Snapshot default output path is updated (see Step 03 for constant change).'
    parallelizable: false
    dependencies:
      - '02-move-scripts'
    next_slice: '02-green'
  - slice_id: '02-green'
    title: 'Verify tree integrity after the move'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - 'rag-index/'
    acceptance_criteria:
      - 'git ls-files scripts/semantic-index/ returns empty.'
      - 'git ls-files rag-index/ lists the moved files.'
      - 'No orphaned paths in jest.config.mjs or package.json yet (full config update in Step 05).'
    parallelizable: false
    dependencies:
      - '02-move-db'
      - '02-move-models'
      - '02-move-proofs'
      - '02-create-snapshots'
```

**User instruction:** Execute this step from the current repo state without relying on prior chat history. Stop and escalate to 00-helping if any downstream consumer breaks or if large artifacts are not copied intact.

**Step objective:** Physically consolidate the RAG surface into `rag-index/`
while preserving the internal tree.

**Stop conditions:** Do not update path constants until the directory move is
clean; do not delete old artifact locations until green validation in Step 08
passes.

---

#### Step 03: Update path constants and internal imports [PLANNED]

```yaml
phase: 1
step: 3
title: 'Update path constants and internal imports'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/rag-update.plans.md'
copy_paste: true
next_step: 'Step 04 — Create rag-index/update-rag.mjs idempotent orchestrator'
owner: '04-implementing'
reviewer: 'Maintainer'
skills:
  - 'implementation-standards'
validation:
  - 'git grep -n "scripts/semantic-index" rag-index/ scripts/agent-customization/gates/cortex-index.gate.mjs scripts/mcp-semantic/tools/cortex-db.mjs .github/skills/repo-cortex-workflow/SKILL.md .github/agents/repo-cortex-scout.agent.md 2>&1 | Select-Object -First 10'
  - 'node -e "import(\'./rag-index/init-schema.mjs\').then(m=>console.log(m.defaultDatabasePath))"'
  - 'node -e "import(\'./rag-index/embed-index.mjs\').then(m=>console.log(m.DEFAULT_MODEL_DIRECTORY))"'
  - 'node -e "import(\'./rag-index/reranker-readiness.mjs\').then(m=>console.log(m.DEFAULT_RERANKER_MODEL_DIRECTORY))"'
acceptance_criteria:
  - 'No runtime file under rag-index/ or the moved consumers references scripts/semantic-index/ or stale data/ paths.'
  - 'DB, model dirs, snapshot output, and hook-context dir resolve to rag-index/ subfolders.'
slices:
  - slice_id: '03-init-schema'
    title: 'Update init-schema.mjs repoRoot and defaultDatabasePath'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - 'rag-index/init-schema.mjs'
    acceptance_criteria:
      - 'repoRoot resolves to repo root from one directory shallower.'
      - 'defaultDatabasePath points to rag-index/data/turso-replica.sqlite.'
    parallelizable: false
    dependencies: []
    next_slice: '03-model-dirs'
  - slice_id: '03-model-dirs'
    title: 'Update model-directory defaults in embed-index.mjs and reranker-readiness.mjs'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - 'rag-index/embed-index.mjs'
      - 'rag-index/reranker-readiness.mjs'
    acceptance_criteria:
      - 'DEFAULT_MODEL_DIRECTORY points to rag-index/models/.'
      - 'DEFAULT_RERANKER_MODEL_DIRECTORY points to rag-index/models/reranker/.'
    parallelizable: false
    dependencies:
      - '03-init-schema'
    next_slice: '03-snapshot-output'
  - slice_id: '03-snapshot-output'
    title: 'Change build-browser-snapshot.mjs default output to rag-index/snapshots/'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - 'rag-index/build-browser-snapshot.mjs'
    acceptance_criteria:
      - 'Default output path is rag-index/snapshots/semantic-snapshot.json.'
      - '--output override still works.'
    parallelizable: false
    dependencies:
      - '03-init-schema'
    next_slice: '03-child-spawns'
  - slice_id: '03-child-spawns'
    title: 'Update child-process spawn paths in prewarm-dense.mjs, session-start-index.mjs, and hooks'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'rag-index/prewarm-dense.mjs'
      - 'rag-index/session-start-index.mjs'
      - 'scripts/agent-customization/hooks/session-start-cortex-mcp-preflight.mjs'
      - 'scripts/agent-customization/hooks/refresh-cortex-after-write.mjs'
    acceptance_criteria:
      - 'All hard-coded script paths point to rag-index/*.mjs.'
    parallelizable: false
    dependencies:
      - '03-model-dirs'
    next_slice: '03-runtime-context'
  - slice_id: '03-runtime-context'
    title: 'Move hook-context directory constant to rag-index/data/hook-context/'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - 'scripts/agent-customization/enforcement/runtime-enforcement.mjs'
    acceptance_criteria:
      - 'RUNTIME_CONTEXT_DIR points to rag-index/data/hook-context/.'
    parallelizable: false
    dependencies:
      - '03-child-spawns'
    next_slice: '03-green'
  - slice_id: '03-green'
    title: 'Verify path constants compile and resolve'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - 'rag-index/'
    acceptance_criteria:
      - 'Node can import init-schema.mjs, embed-index.mjs, and reranker-readiness.mjs and print the new default paths.'
      - 'git grep shows no scripts/semantic-index strings in the moved scripts.'
    parallelizable: false
    dependencies:
      - '03-snapshot-output'
      - '03-child-spawns'
      - '03-runtime-context'
```

**User instruction:** Execute this step from the current repo state without relying on prior chat history. Stop and escalate to 00-helping if any downstream consumer breaks or if large artifacts are not copied intact.

**Step objective:** Make every script resolve its new home correctly after the
move.

---

#### Step 04: Create rag-index/update-rag.mjs idempotent orchestrator [PLANNED]

```yaml
phase: 1
step: 4
title: 'Create rag-index/update-rag.mjs idempotent orchestrator'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/rag-update.plans.md'
copy_paste: true
next_step: 'Step 05 — Update package.json, jest.config.mjs, and .gitignore'
owner: '04-implementing'
reviewer: 'Maintainer'
skills:
  - 'implementation-standards'
  - 'red-test-contracts'
validation:
  - 'node rag-index/update-rag.mjs --dry-run --json'
  - 'node rag-index/update-rag.mjs --json'
  - 'node rag-index/update-rag.mjs --dry-run --validate --json'
acceptance_criteria:
  - 'The script runs from repo root and exits 0 on a healthy corpus.'
  - '--dry-run lists stages without mutating DB/models/snapshot.'
  - 'Second consecutive run on an unchanged corpus reports skip/zero work.'
  - '--validate runs the cortex-index gate and fails on red results.'
  - 'JSON summary contains a stages array with name/status/elapsedMs.'
slices:
  - slice_id: '04-red'
    title: 'Write red tests for update-rag.mjs CLI behavior'
    status: '[PLANNED]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'rag-index/__tests__/update-rag.test.mjs'
    acceptance_criteria:
      - 'Red tests fail before the orchestrator is implemented.'
      - 'Tests cover --dry-run, --validate, stage ordering, and idempotency claims.'
    parallelizable: false
    dependencies: []
    next_slice: '04-impl'
  - slice_id: '04-impl'
    title: 'Implement update-rag.mjs stage orchestrator'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'rag-index/update-rag.mjs'
    acceptance_criteria:
      - 'Chains build → prewarm/embed → build-terms → build-graph → snapshot → optional validation in order.'
      - 'Skips graph build when no corpus changes are detected (freshness check).'
      - 'Reports per-stage status and a machine-readable JSON summary.'
    parallelizable: false
    dependencies:
      - '04-red'
    next_slice: '04-green'
  - slice_id: '04-green'
    title: 'Green validation of update-rag.mjs'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'rag-index/__tests__/update-rag.test.mjs'
    acceptance_criteria:
      - 'All red tests pass.'
      - '--dry-run and --validate behave as specified.'
    parallelizable: false
    dependencies:
      - '04-impl'
```

**User instruction:** Execute this step from the current repo state without relying on prior chat history. Stop and escalate to 00-helping if any downstream consumer breaks or if large artifacts are not copied intact.

**Step objective:** Provide a single, safe, incremental entry point for the
entire RAG pipeline.

**Stage ordering:**

1. Build/refresh corpus (`rag-index/build-index.mjs`).
2. Ensure dense model + embeddings (`rag-index/prewarm-dense.mjs`).
3. Build term embeddings (`rag-index/build-term-index.mjs`).
4. Build entity graph (`rag-index/build-entity-graph.mjs`) — skipped when the
   corpus is unchanged; when run, it performs a full rebuild.
5. Build browser snapshot (`rag-index/build-browser-snapshot.mjs`).
6. Optional validation (`cortex-index` gate and/or `validate-index.mjs`).

---

#### Step 05: Update package.json, jest.config.mjs, and .gitignore [PLANNED]

```yaml
phase: 1
step: 5
title: 'Update package.json, jest.config.mjs, and .gitignore'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/rag-update.plans.md'
copy_paste: true
next_step: 'Step 06 — Update MCP wiring, gates, hooks, skills, and agents'
owner: '04-implementing'
reviewer: 'Maintainer'
skills:
  - 'implementation-standards'
validation:
  - 'git grep -n "scripts/semantic-index" package.json jest.config.mjs .gitignore'
  - 'npm run rag:update -- --dry-run'
acceptance_criteria:
  - 'All RAG npm scripts point to rag-index/ entry points.'
  - 'jest.config.mjs testMatch references rag-index/ instead of scripts/semantic-index/.'
  - '.gitignore ignores new generated paths and no longer ignores stale old paths.'
  - 'A new rag:update script invokes the unified orchestrator.'
slices:
  - slice_id: '05-package-json'
    title: 'Repoint RAG npm scripts and add rag:update'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'package.json'
    acceptance_criteria:
      - 'No package.json script references scripts/semantic-index/.'
      - '"rag:update": "node rag-index/update-rag.mjs" exists.'
      - 'Existing index:* names are preserved with repointed paths.'
    parallelizable: false
    dependencies: []
    next_slice: '05-jest'
  - slice_id: '05-jest'
    title: 'Rename jest.config.mjs semantic-index projects to rag-index'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - 'jest.config.mjs'
    acceptance_criteria:
      - 'Projects semantic-index-scripts and semantic-index-mjs become rag-index-scripts and rag-index-mjs.'
      - 'testMatch globs use rag-index/**/*.test.ts and rag-index/**/*.test.mjs.'
    parallelizable: false
    dependencies:
      - '05-package-json'
    next_slice: '05-gitignore'
  - slice_id: '05-gitignore'
    title: 'Update .gitignore for new generated artifact paths'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - '.gitignore'
    acceptance_criteria:
      - 'Ignores rag-index/data/*.sqlite, rag-index/data/hook-context/*.json, rag-index/models/, rag-index/freshness-proofs/*/, rag-index/snapshots/semantic-snapshot.json, rag-index/data/mcp-session-override.json.'
      - 'Removes or narrows stale entries for data/turso-replica.sqlite, data/hook-context-*.json, scripts/semantic-index/models/, root freshness-proof-*/.'
    parallelizable: false
    dependencies:
      - '05-jest'
    next_slice: '05-green'
  - slice_id: '05-green'
    title: 'Verify npm scripts and jest project discovery'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - 'package.json'
      - 'jest.config.mjs'
    acceptance_criteria:
      - 'npm run rag:update -- --help exits 0.'
      - 'npx jest --config=jest.config.mjs --listTests --selectProjects rag-index-scripts --selectProjects rag-index-mjs lists the moved tests.'
    parallelizable: false
    dependencies:
      - '05-gitignore'
```

**User instruction:** Execute this step from the current repo state without relying on prior chat history. Stop and escalate to 00-helping if any downstream consumer breaks or if large artifacts are not copied intact.

**Step objective:** Make the repository commands and test discovery reflect the
new layout.

---

#### Step 06: Update MCP wiring, gates, hooks, skills, and agents [PLANNED]

```yaml
phase: 1
step: 6
title: 'Update MCP wiring, gates, hooks, skills, and agents'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/rag-update.plans.md'
copy_paste: true
next_step: 'Step 07 — Clean stale temporal artifacts and register the plan'
owner: '04-implementing'
reviewer: 'Maintainer'
skills:
  - 'implementation-standards'
  - 'mcp-local-server-workflow'
validation:
  - 'git grep -n "scripts/semantic-index\|data/turso-replica.sqlite\|docs/assets/semantic-snapshot.json" scripts/mcp-semantic/ scripts/agent-customization/gates/cortex-index.gate.mjs .github/skills/repo-cortex-workflow/SKILL.md .github/agents/repo-cortex-scout.agent.md .vscode/mcp.json .mcp.json'
  - 'neataptic-gate-mcp:run_gate_check --gate=cortex-index'
acceptance_criteria:
  - 'MCP server config points to rag-index/data/turso-replica.sqlite.'
  - 'cortex-db.mjs, cortex-index.gate.mjs, and skill/agent docs reference rag-index/ paths.'
  - 'cortex-index gate passes after the updates.'
slices:
  - slice_id: '06-mcp-config'
    title: 'Update .vscode/mcp.json and .mcp.json cortex DB URL'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - '.vscode/mcp.json'
      - '.mcp.json'
    acceptance_criteria:
      - 'TURSO_DATABASE_URL points to file:./rag-index/data/turso-replica.sqlite.'
    parallelizable: false
    dependencies: []
    next_slice: '06-mcp-tools'
  - slice_id: '06-mcp-tools'
    title: 'Update mcp-semantic imports and default path resolution'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'scripts/mcp-semantic/tools/cortex-db.mjs'
      - 'scripts/mcp-semantic/repo-cortex-mcp.mjs'
    acceptance_criteria:
      - 'Imports point to rag-index/ modules.'
      - 'Fix hints point to rag-index/ scripts.'
    parallelizable: false
    dependencies:
      - '06-mcp-config'
    next_slice: '06-gates'
  - slice_id: '06-gates'
    title: 'Update agent-customization gates and fix hints'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'scripts/agent-customization/gates/cortex-index.gate.mjs'
      - 'scripts/agent-customization/gates/cortex-embeddings.gate.mjs'
      - 'scripts/agent-customization/gates/cortex-first-search.gate.mjs'
    acceptance_criteria:
      - 'Default snapshot path points to rag-index/snapshots/semantic-snapshot.json.'
      - 'All fix hints and imports reference rag-index/ paths.'
    parallelizable: false
    dependencies:
      - '06-mcp-tools'
    next_slice: '06-skills-agents'
  - slice_id: '06-skills-agents'
    title: 'Update repo-cortex-workflow skill and repo-cortex-scout agent examples'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - '.github/skills/repo-cortex-workflow/SKILL.md'
      - '.github/agents/repo-cortex-scout.agent.md'
    acceptance_criteria:
      - 'No stale scripts/semantic-index/ or data/turso-replica.sqlite examples remain.'
      - 'Snapshot examples point to rag-index/snapshots/semantic-snapshot.json.'
    parallelizable: false
    dependencies:
      - '06-gates'
    next_slice: '06-green'
  - slice_id: '06-green'
    title: 'Verify MCP/gates still pass with new paths'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'scripts/mcp-semantic/tools/cortex-db.mjs'
      - 'scripts/agent-customization/gates/cortex-index.gate.mjs'
    acceptance_criteria:
      - 'neataptic-gate-mcp:run_gate_check --gate=cortex-index passes.'
      - 'A live search_corpus query returns non-empty results.'
    parallelizable: false
    dependencies:
      - '06-skills-agents'
```

**User instruction:** Execute this step from the current repo state without relying on prior chat history. Stop and escalate to 00-helping if any downstream consumer breaks or if large artifacts are not copied intact.

**Step objective:** Keep the live Repo Cortex MCP server and its diagnostics
pointed at the relocated index.

---

#### Step 07: Clean stale temporal artifacts and register the plan [PLANNED]

```yaml
phase: 1
step: 7
title: 'Clean stale temporal artifacts and register the plan'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/rag-update.plans.md'
copy_paste: true
next_step: 'Step 08 — Green validation'
owner: '04-implementing'
reviewer: 'Maintainer'
skills:
  - 'implementation-standards'
  - 'plan-sync-validation'
  - 'tracker-handoff'
validation:
  - 'git grep -n "scripts/semantic-index\|data/turso-replica.sqlite\|freshness-proof-" -- "*.mjs" "*.ts" "*.md" "package.json" ".gitignore" "jest.config.mjs"'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/rag-update.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-sync'
acceptance_criteria:
  - 'Old artifact locations are empty or removed.'
  - 'plans/README.md and plans/Roadmap.md reference this plan.'
  - 'No tracked runtime file references the old layout.'
slices:
  - slice_id: '07-remove-old-tree'
    title: 'Remove empty scripts/semantic-index/ directory and stale root artifacts'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - 'scripts/semantic-index/'
      - 'data/turso-replica.sqlite'
      - 'data/hook-context-*.json'
      - 'data/mcp-session-override.json'
      - 'freshness-proof-*/'
      - 'missing-semantic-index.sqlite'
    acceptance_criteria:
      - 'scripts/semantic-index/ no longer exists.'
      - 'Old data/ sqlite and hook contexts are gone (eval-baselines/ remains).'
      - 'Root freshness-proof-* directories and missing-semantic-index.sqlite are gone.'
    parallelizable: false
    dependencies: []
    next_slice: '07-register-plan'
  - slice_id: '07-register-plan'
    title: 'Register plan in plans/README.md and plans/Roadmap.md'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 1
    files_to_change:
      - 'plans/README.md'
      - 'plans/Roadmap.md'
    acceptance_criteria:
      - 'README.md links this plan with trigger phrases (rag index, semantic index reorganization, rag-index consolidation).'
      - 'Roadmap.md contains a standalone meta-workflow/infrastructure lane for this plan.'
    parallelizable: false
    dependencies:
      - '07-remove-old-tree'
    next_slice: '07-green'
  - slice_id: '07-green'
    title: 'Verify no stale references remain and plan-sync passes'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - 'plans/README.md'
      - 'plans/Roadmap.md'
    acceptance_criteria:
      - 'git grep for old paths returns no runtime hits.'
      - 'validate-plan-sync passes.'
    parallelizable: false
    dependencies:
      - '07-register-plan'
```

**User instruction:** Execute this step from the current repo state without relying on prior chat history. Stop and escalate to 00-helping if any downstream consumer breaks or if large artifacts are not copied intact.

**Step objective:** Complete the no-deferred-cleanup requirement and make the
plan discoverable.

---

#### Step 08: Green validation [PLANNED]

```yaml
phase: 1
step: 8
title: 'Green validation'
status: '[PLANNED]'
goal: 'green-testing'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/rag-update.plans.md'
copy_paste: true
next_step: 'Compress Phase 1 and archive to plans/completed/'
owner: '05-green-testing'
reviewer: 'Maintainer'
skills:
  - 'green-validation-gates'
  - 'coverage-guard'
validation:
  - 'npm run rag:update -- --dry-run'
  - 'npm run rag:update -- --validate --json'
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects rag-index-scripts --selectProjects rag-index-mjs --runInBand'
  - 'npm run lint'
  - 'neataptic-gate-mcp:run_gate_check --gate=cortex-index'
  - 'neataptic-gate-mcp:run_gate_check --gate=plan-sync'
acceptance_criteria:
  - 'npm run rag:update -- --validate --json exits 0 and reports all stages ok/skipped.'
  - 'RAG jest projects pass with zero failures.'
  - 'npm run lint passes on touched files.'
  - 'cortex-index and plan-sync gates pass.'
```

**User instruction:** Execute this step from the current repo state without relying on prior chat history. Stop and escalate to 00-helping if any downstream consumer breaks or if large artifacts are not copied intact.

**Step objective:** Prove the relocated RAG stack is healthy end-to-end.

---

## Validation gates

- `validate-plan-sync` passes for `plans/rag-update.plans.md`.
- `validate-plan-phase-packets` passes for `plans/rag-update.plans.md`.
- `cortex-index` gate passes after relocation.
- `npm run rag:update -- --dry-run` and `--validate --json` run without errors.
- RAG-focused jest projects and `npm run lint` are green.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Workstream: RAG/index infrastructure reorganization for NeatapticTS.
Current boundary: Phase 1 Step 01 [DONE]; the plan document at plans/rag-update.plans.md has been authored and decision records are recorded.
Already covered:
- Catalogued scripts/semantic-index/ contents, package.json scripts, .gitignore, and external path references.
- Delegated boundary mapping, risk assessment, and acceptance-criteria drafting to the appropriate specialists.
- Decided on a flat rename strategy for rag-index/ (preserving internal subfolders), snapshot output moving to rag-index/snapshots/, and wiring build-entity-graph.mjs as a conditional full rebuild.

Next narrow task:
- Execute Step 02: move scripts/semantic-index/ → rag-index/ and relocate generated artifacts (DB, models, hook contexts, freshness proofs, snapshots) into rag-index/ subfolders.

Required validations before advancing:
- git ls-files scripts/semantic-index/ returns empty.
- git ls-files rag-index/ lists the moved files.
- Large artifacts (DB, models) copied intact; do not delete old copies until Step 08 green validation passes.

Cautions:
- Do not update path constants until the physical move is clean.
- Keep data/eval-baselines/ in place; do not move it.
- The live cortex MCP server in .vscode/mcp.json must be updated in Step 06, not before the DB is in its new location.
```
