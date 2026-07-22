# Cortex Orchestration Single Source of Truth

**Status:** [DONE] · **Plan ID:** CORTEX_ORCHESTRATION_SSOT
**Downstream of:** `plans/completed/Repo_Cortex_Advanced_RAG_Architecture.plans.md` [DONE], `plans/completed/MCP_Lazy_Load_Facade.plans.md` [DONE]

## Thesis

The Repo Cortex (Turso/libSQL RAG) is already capable of `search_corpus`, `search_context`, `search_advanced`, `traverse_graph`, `expand_query`, and `submit_feedback`, with hybrid BM25 + dense + RRF ranking, native DiskANN vectors, metadata filtering, and a lazy-load facade. Plan chunks are already indexed under family `plan` / `completed-plan`. Yet orchestration agents still read `.plans.md` and `.research` files directly via `read_file`, and execution specialists receive only the slice packet text the orchestrator chooses to paste — not an assembled Cortex context window.

This plan closes that gap in five incremental phases:

- **Phase A — Slice metadata enrichment.** Add `slice_id`, `step_number`, `phase`, `status` columns to the `chunks` table; enrich plan chunks during embedding; expose `slice_id` / `step_number` as filter parameters in `search_context` and `search_corpus`.
- **Phase B — `get_slice_context` workflow tool.** Add a new tool to `neataptic-workflow-mcp.mjs` that calls `search_context` with the active slice metadata and returns an assembled context window (step packet + relevant source + boundary notes + related test contracts). Expose it through the lazy facade and add it to Tier-1 agent tool lists.
- **Phase C — Auto-reindex hook.** Add a post-commit hook (and an optional dev-mode file watcher) that detects changes to `plans/*.plans.md` and triggers a targeted `embed-index.mjs --files=<changed>` re-index of just the changed plan files. Extend `embed-index.mjs` to support `--files`, update the `cortex-index` gate's fixHint to prefer the targeted path, and log re-index results so freshness checks see an up-to-date timestamp. This keeps the index current while agents constantly edit plans during orchestration.
- **Phase D — Pre-execute hook convention.** Add a `pre_execute_hook` field to the step-packet convention. When `04-implementing` dispatches a specialist, the packet includes `{ pre_execute_hook: { tool: "neataptic-workflow-mcp/get_slice_context", args: { slice_id: "..." } } }` and the specialist calls it before reading any files.
- **Phase E — Pull-to-push migration.** Update `02-researching`, `04-implementing`, and specialist agent bodies to prefer `get_slice_context` / `search_context` over direct `read_file` of plan/research files. Keep `read_file` as a fallback only. Update the `research-methodology` skill's Cortex-First policy to mention slice-aware retrieval.

The end state: agents query the Cortex MCP directly for orchestration context, execution sub-agents receive rich slice context automatically via a pre-execute hook instead of relying on the orchestrator to paste the right excerpts, and the RAG index stays fresh automatically as plans evolve.

## Final state

[DONE] — Phases A, B, C, D, and E are complete and compressed to `plans/completed/Cortex_Orchestration_Single_Source_of_Truth.logs.md`. All Phase E steps (D1, D2, D3) and slices are green-validated. Targeted RAG re-index, step-packet gate, and plan-sync gate all pass. Plan archived by `07-logging`.

- Step D1 — Migrate 04-implementing and specialists to `get_slice_context` [DONE]
- Step D2 — Update `research-methodology` skill and 02-researching agent [DONE]
- Step D3 — Final green validation of Phase E [DONE]

## Latest validation evidence

```text
green-light: true
status: Phase E is [DONE] and compressed to `plans/completed/Cortex_Orchestration_Single_Source_of_Truth.logs.md`. Targeted RAG re-index, step-packet gate, and plan-sync gate pass after compression.
verification timestamp: 2026-07-20T22:08-04:00
verifier: 07-logging
findings:
- 'Phase E Step D3 green validation complete: all 7 red tests pass and all declared gates pass.'
- 'node rag-index/build-index.mjs --files=plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md: scanned 1, indexed 1, skipped 0, chunks 29'
- 'node rag-index/embed-index.mjs --files=plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md: embedded 29, skipped 23686'
- 'step-packet gate: pass'
- 'plan-sync gate: pass'
- 'Sample marker heading verified exactly once in the plan file.'
  next: 'Plan archived to plans/completed/.'
```

## Non-goals

- This plan does **NOT** change core NEAT algorithms in `src/neat/` or any evolutionary correctness surface.
- This plan does **NOT** replace the file-based plan authoring surface — humans still write and edit `.plans.md` files directly. The Cortex indexes them; it does not own authoring.
- This plan does **NOT** remove `read_file` from agents. `read_file` remains a fallback for degraded Cortex, for files outside the corpus, and for ad-hoc inspection.
- This plan does **NOT** change the Turso cloud topology, embedded replica layout, or DiskANN configuration.
- This plan does **NOT** introduce a new MCP server. It adds a tool to the existing `neataptic-workflow-mcp` and routes it through the existing lazy facade.
- This plan does **NOT** migrate NeatChat's local retrieval/memory DB — that is a separate surface owned by `plans/completed/NeatChat_Local_Retrieval_Memory.plans.md`.

## No deferred cleanup

Any slice that migrates agents from `read_file` to `get_slice_context` MUST remove the old direct-read patterns in the same slice. No dual-path compatibility wrappers, no "use Cortex OR read_file" branches left in agent bodies as a permanent state. `read_file` remains available as a fallback only in the explicit degraded-Cortex path documented in `research-methodology`. A slice that introduces `get_slice_context` calls alongside unchanged `read_file`-first prose in the same agent body is a planning defect and MUST be rejected before implementation.

## Dependencies

- **Downstream of:** `plans/completed/Repo_Cortex_Advanced_RAG_Architecture.plans.md` [DONE] — provides `search_context`, metadata filtering, hybrid ranking, and the enriched `chunks` table this plan extends.
- **Downstream of:** `plans/completed/MCP_Lazy_Load_Facade.plans.md` [DONE] — provides the single `cortex` router tool facade at `scripts/agent-customization/mcp/cortex-tier-tool.mjs` that this plan extends with `get_slice_context`.
- **Soft dependency:** `plans/completed/Step_Packet_Goal_Redesign.plans.md` [DONE] — the `goal`-based step packet shape is the surface Phase D extends with `pre_execute_hook`.
- **Soft dependency:** `plans/mcp-active-binding.plans.md` [WIP] — keeps `neataptic-workflow-mcp` perpetually bound so the new `get_slice_context` tool is always discoverable.

## Implementation phases

### Phase A — Slice metadata enrichment [DONE]

**Goal:** Add structured step/slice metadata to plan chunks so `search_context` can filter by `slice_id` and `step_number`.

[DONE] Phase A completed. Detailed step/slice records, validation evidence, and fix cycles are in `plans/Cortex_Orchestration_Single_Source_of_Truth.logs.md`.

- Step A1: schema migration and embed-index slice metadata enrichment — 100% coverage, plan gates pass.
- Step A2: `search_context` / `search_corpus` `slice_id` and `step_number` filter parameters — 100% coverage on touched `scripts/mcp-semantic/tools/` files, 250/250 targeted tests pass.
- Step A3-green: Phase A final validation — all targeted suites green, coverage gate passes, `build-index` OK; `cortex-index` gate blocked only by `workflow_mcp_alive=false` (MCP infrastructure issue, documented).

### Phase B — get_slice_context workflow tool [DONE]

**Goal:** Add a `get_slice_context` tool to `neataptic-workflow-mcp` that returns an assembled context window for a given `slice_id`.

[DONE] Phase B completed. Detailed step/slice records, validation evidence, and gate exception records are in `plans/Cortex_Orchestration_Single_Source_of_Truth.logs.md`.

- Step B1-red-tests / B1-tool-impl / B1-green: `get_slice_context` registration, context assembly, and green validation.
- Slice B2-facade-and-tools: `createSliceContextTool` added to `cortex-tier-tool.mjs`, registered in `neataptic-gate-mcp.mjs`, exposed via `cortex-facade.mjs` with lazy-facade local-tool support, mirrored in the lazy-facade comparison fixture, and listed in all eight Tier-1 agent frontmatter files.
- B3-green final validation: 153/153 targeted tests pass; plan-sync, step-packet, plan-slice-quality, and validate-plan-sync gates pass; scoped-baseline code-coverage gate passes. Three repo-wide gates failed with documented exceptions: `cortex-index` (`workflow_mcp_alive=false`, passes with `--plan=plans/mcp-active-binding.plans.md`), `devtools-coverage` (03/05 agents missing `devtools` skill), and `specialist-review` (unrelated `Agentic_Workflow_Architecture.plans.md` missing `VALIDATION_EVIDENCE`). Exceptions captured in `.github/ai-learning/learning-log.jsonl`.

### Phase C — Auto-reindex hook [DONE]

**Goal:** Keep the Cortex RAG index fresh automatically when `plans/*.plans.md` files change, by adding targeted `--files` re-index support to `embed-index.mjs` and wiring a post-commit hook (plus an optional dev-mode file watcher) that triggers it.

[DONE] Phase C completed. Detailed step/slice records, validation evidence, fix cycles, and the final E2-green PlanUpdate are in `plans/Cortex_Orchestration_Single_Source_of_Truth.logs.md`.

- Step E1: `--files` targeted re-index support in `rag-index/embed-index.mjs` and `rag-index/cli-utils.mjs` — red tests, implementation, and green validation passed; scoped coverage baselines recorded.
- Step E2: post-commit hook (`rag-index/git-hooks/post-commit`), `rag-index/auto-reindex.mjs`, `rag-index/watch-plans.mjs`, repeatable `--files` support in `rag-index/build-index.mjs`, and `cortex-index.gate.mjs` fixHint update — red tests, implementation (two specialist fix cycles), and green validation passed.
- Plan gates: plan-sync, step-packet, plan-slice-quality, validate-plan-sync all pass.
- Index tooling: targeted `build-index.mjs --files=...` and `embed-index.mjs --files=...` re-index verified; post-commit hook staged with executable bit (`100755`).

### Phase D — Pre-execute hook convention [DONE]

**Goal:** Add a `pre_execute_hook` field to the step-packet convention so specialists receive a declared Cortex query to run before reading any files.

[DONE] Phase D — Pre-execute hook convention. Added `pre_execute_hook` to step-packet gate, documented it in phase-handoff-workflow skill, updated 04-implementing agent body to honor the hook, and added a sample packet using `get_slice_context`. All slices (C1-red-tests, C1-gate-and-skill, C1-green, C2-red-tests, C2-agent-and-sample, C3-green) passed; 36/36 targeted gate tests green; plan-slice-quality, step-packet, agent-graph, plan-sync, routing-table-freshness, cortex-index, and code-coverage gates pass. Details moved to `plans/Cortex_Orchestration_Single_Source_of_Truth.logs.md`.

### Phase E — Pull-to-push migration [DONE]

**Goal:** Migrate `02-researching`, `04-implementing`, and specialist agent bodies to prefer `get_slice_context` / `search_context` over direct `read_file` of plan/research files. Update the `research-methodology` skill's Cortex-First policy to mention slice-aware retrieval. Old direct-read patterns are removed in the same slice — no dual-path.

[DONE] Phase E completed. Detailed step/slice records, validation evidence, PlanUpdate blocks, and the phase packet are in `plans/Cortex_Orchestration_Single_Source_of_Truth.logs.md`.

- Step D1 — D1-pull-to-push: Migrate 04-implementing specialists to get_slice_context [DONE]
- Step D2 — D2-research-skill: Update research-methodology skill and 02-researching agent [DONE]
- Step D3 — D3-green: Final validation of Phase E [DONE]

Sample step packet using the new convention (referenced by AC-D004 / AC-D1-001):

```yaml
phase: E
step: 1
title: 'Migrate 04-implementing specialists to get_slice_context'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md'
copy_paste: true
pre_execute_hook:
  tool: 'neataptic-workflow-mcp/get_slice_context'
  args:
    slice_id: 'D1-impl'
next_step: 'Step D2 — Update research-methodology skill and 02-researching agent'
skills:
  - 'implementation-standards'
  - 'repo-cortex-workflow'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
acceptance_criteria:
  - id: AC-D1-SAMPLE
    text: 'Sample packet demonstrates pre_execute_hook pointing at neataptic-workflow-mcp/get_slice_context'
    validation: 'neataptic-gate-mcp:run_gate_check --gate=step-packet'
```

---

## Validation gates

Required validation gates for this plan:

- `plan-slice-quality`: passes when all WIP slices are within the 4-hour estimate limit.
- `step-packet`: passes when all active WIP phase/step YAML blocks conform to the new format.
- `plan-sync`: passes when the plan is correctly registered in `plans/README.md` and `plans/Roadmap.md`.
- `agent-graph`: passes when the agent frontmatter graph is consistent (run after agent-body changes).
- `cortex-index`: passes when the Repo Cortex index is fresh for this plan.

---

## Audit log

```text
PlanUpdate:
  slice_id: C1-gate-and-skill
  changed_files:
    - scripts/agent-customization/gates/step-packet.gate.mjs
    - scripts/agent-customization/customization-utils.mjs
    - coverage/coverage-baseline.json
    - plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json: OK'
    - 'npm run lint: 0 errors, 29 pre-existing warnings'
    - 'npx prettier --check scripts/agent-customization/gates/step-packet.gate.mjs scripts/agent-customization/customization-utils.mjs coverage/coverage-baseline.json: all formatted'
    - 'node scripts/agent-customization/gates/step-packet.gate.mjs --json: pass, 0 violations'
  gate_checks:
    - 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality: pass'
    - 'neataptic-gate-mcp:run_gate_check --gate=step-packet: pass'
    - 'neataptic-gate-mcp:run_gate_check --gate=plan-sync: pass'
    - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md: pass'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=scripts/agent-customization/gates/step-packet'
  rollback:
    - 'git checkout -- scripts/agent-customization/gates/step-packet.gate.mjs'
    - 'git checkout -- scripts/agent-customization/customization-utils.mjs'
    - 'git checkout -- coverage/coverage-baseline.json'
    - 'git checkout -- plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md'
  next: 'Hand off to 05-green-testing to run the focused jest slice and coverage-guard for AC-C1-003'
```

```text
PlanUpdate:
  slice_id: D1-impl-fix
  changed_files:
    - plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md
    - .github/agent-skill-routing-table.md
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json: OK'
    - 'npm run lint: 0 errors, 29 pre-existing warnings'
    - 'npx prettier --check plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md .github/agent-skill-routing-table.md: all formatted'
  gate_checks:
    - 'routing-table-freshness gate: pass (table hash matches source frontmatter)'
    - 'agent-graph gate: pass (67 agents, 0 issues)'
    - 'step-packet gate: pass (4 plans scanned, 0 violations)'
    - 'workflow-update-sync.mjs: pass (between-steps)'
    - 'validate-plan-sync.mjs: pass (0 errors, 0 warnings)'
  tests_for_green:
    - 'None — this slice only fixes plan metadata and regenerates the routing table.'
  rollback:
    - 'git checkout -- plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md'
    - 'git checkout -- .github/agent-skill-routing-table.md'
  next: 'Run requested gate checks; if all pass, hand off to 05-green-testing for any pending D1-impl tests.'
```

```text
PlanUpdate:
  slice_id: D1-header-format-fix
  changed_files:
    - plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md
  preflight:
    - 'npx prettier --check plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md: all formatted'
  gate_checks:
    - 'neataptic-gate-mcp:run_gate_check --gate=step-packet: pass'
    - 'neataptic-gate-mcp:run_gate_check --gate=plan-slice-quality: pass'
    - 'neataptic-gate-mcp:run_gate_check --gate=plan-sync: pass'
    - 'neataptic-gate-mcp:run_gate_check --gate=cortex-index: pass (after re-index)'
    - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md: pass'
    - 'node scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs --plan=plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md --self-check --json: PASS, snapshot step=D1, agent=implementing'
  tests_for_green:
    - 'None — plan-format fix only.'
  rollback:
    - 'git checkout -- plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md'
  next: 'Restart IDE/Copilot session to respawn neataptic-workflow-mcp stdio server, then call get_active_workflow_snapshot to confirm activeStep.number === "D1".'
```
