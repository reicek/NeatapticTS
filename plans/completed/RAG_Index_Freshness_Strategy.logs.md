# RAG Index Freshness Strategy — Log

**Status:** [DONE] — Plan complete. Phase 0 re-validated GREEN 2026-08-26; Phase 1 re-validated GREEN 2026-08-26; Phase 2 independently GREEN 2026-08-27; Phase 3 (P3-S1-A + P3-S1-B) independently GREEN 2026-08-27. This log holds the compressed done-state records; the plan file carries compact [DONE] markers only.

## Incident: CLI task-tool registry corruption and recovery

### Root cause

The Copilot CLI `task` tool was rejecting every numbered Tier-1 orchestrator except `04-implementing` because **seven of the eight Tier-1 `.agent.md` files were encoded as Windows-1252 instead of UTF-8**. The CLI parser reads these files as UTF-8, hits invalid byte sequences (Windows-1252 em-dashes such as `0x97`), and silently drops the agent from the registry.

Files that were invalid UTF-8 and unreachable:

- `00-helping.agent.md`
- `01-planning.agent.md`
- `02-researching.agent.md`
- `03-red-testing.agent.md`
- `05-green-testing.agent.md`
- `06-documenting.agent.md`
- `07-logging.agent.md`
- `implementation-executor.agent.md`

A second, smaller issue: `.github/agents/templates/` and `.github/agents/examples/` contained orphan markdown files (`boundary-recon.md`, `prior-art-scan.md`, `structured-v1-example.md`). Nothing in the repo references those files, yet the CLI scanner recursed into `.github/agents/` and exposed them as bogus agent types.

### Fixes applied

1. Converted the 8 invalid agent files to valid UTF-8 by decoding as Windows-1252 and re-encoding as UTF-8. Content preserved; only the byte encoding of em-dashes/en-dashes corrected.
2. Moved orphan subdirectories out of `.github/agents/` so the CLI scanner can no longer misidentify them as agents.
3. Regenerated `.github/agent-skill-routing-table.md` (38 agents, 67 skills) and re-ran repo-side gates:
   - `npm run agents:routing-table`: regenerated
   - `routing-table-freshness.gate.mjs`: PASS
   - `tier-enforcement.gate.mjs`: PASS (8 Tier-1, 3 Tier-2, 26 Tier-3, 1 Tier-4, 8 user-invocable)
   - `agent-graph.gate.mjs`: PASS
   - `validate-agent-frontmatter.mjs`: 0 errors, 0 warnings
   - `validate-skill-frontmatter.mjs`: 0 errors, 0 warnings

### Preventive validation added

- Added `readWorkspaceFileBytes()` and `isValidUtf8()` to `scripts/agent-customization/customization-utils.mjs`.
- `validate-agent-frontmatter.mjs` now rejects any `.agent.md` file that is not valid UTF-8 before parsing frontmatter.

### Re-start protocol

1. Restart the Copilot CLI / VS Code window to flush the stale task-tool cache.
2. Verify orchestrator reachability with a dry dispatch of `02-researching` and `05-green-testing`.
3. Run repo-side gate baseline.
4. Phase 0 re-validation: dispatch fresh `02-researching` to audit Phase 0, then `05-green-testing` to green-validate.
5. Phase 1 re-validation: only after Phase 0 is green, dispatch fresh `02-researching` to audit Phase 1, then `05-green-testing` to green-validate.
6. Do not enter Phase 2 until both phases pass independent agent validation.

## Phase 0 — The fix: implementation complete

[DONE] Step 01 — Remove age gate, add deletion sweep, per-family manifest and fresh emission, gate `family_fresh` map.
[DONE] Step 02 — Session-start touch pass simplification and README docs for new freshness model.

### Slices completed

- **P0-S1-A:** Remove age gate + add deletion sweep + update type declarations.
- **P0-S1-B:** Per-family `fresh` emission + manifest + atomic writes.
- **P0-S1-C:** Gate `family_fresh` map + runtime fixHint + freshness hook.
- **P0-S1-D:** Pre-dispatch freshness hook (read-only) + skill docs.
- **P0-S1-E:** Existing test fixture cleanup (removed API references).
- **P0-S2-A:** Simplify `session-start-index.mjs` touch pass.
- **P0-S2-B:** README docs for new freshness model.

### Key shell-level validation (produced before agent restoration)

- `grep` for legacy age-gate/manifest strings returned no hits.
- Jest validate-index suites: 27 passed (2 suites).
- Jest cortex-index.gate suites: 55 passed (3 suites).
- Jest pre-dispatch-freshness-hook: 34 passed (1 suite).
- Jest session-start-index: 10 passed (1 suite).
- `node rag-index/validate-index.mjs --json`: ok=true, pass=true, all 10 families fresh.
- `npx tsc --noEmit -p tsconfig.json`: exit 0.
- `npm run lint`: exit 0.

### Removed files

- `artifacts/cortex-freshness-manifest.json`

## Fix packet P0-AC012-iteration-1 — RESOLVED

Removed remaining literal references to the age gate and legacy manifest so the AC-012 grep returned no hits.

## Phase 1 — Hook extension: implementation complete

[DONE] Step 01 — Extend post-commit and watcher globs to cover `src/**/*.ts`, `.github/skills/**/*.md`, `.github/agents/**/*.md`, and `.github/copilot-instructions.md`.

### Slices completed

- **P1-S1-A:** Extend post-commit hook glob.
- **P1-S1-B:** Extend watcher globs.

### Key shell-level validation (produced before agent restoration)

- Jest auto-reindex: 15 passed (2 suites).
- Jest freshness-hooks: 55 passed (2 suites).
- `node rag-index/validate-index.mjs --json --min-documents 1 --min-chunks 1`: ok=true, pass=true, all 10 family_fresh families report fresh.
- `npx tsc --noEmit -p tsconfig.json`: exit 0.
- `npm run lint -- --max-warnings=0`: exit 0.

## Current blocker — RESOLVED 2026-08-26

The CLI task-tool registry corruption was fixed (UTF-8 agent files; orphan dirs moved out of `.github/agents/`), a fresh session dispatched `00/02/04/05` successfully, and Phase 0 was independently re-validated GREEN on 2026-08-26 (see "Phase 0 — Independent re-validation frontier" below). **Phase 1 re-validation is the active frontier**; Phase 2 remains gated on Phase 1 passing independent agent validation.

## Plan compression note

The verbose per-slice `PlanUpdate` evidence that previously lived in `plans/RAG_Index_Freshness_Strategy.plans.md` under `## Latest validation evidence` has been compressed into this log. The plan file now carries only the active frontier, restart protocol, and a `## Handoff query` for the next fresh CLI session.

## Phase 0 — Full detail (moved from plan tracker, 2026-08-26)

Per the step-compression policy, the remaining Phase 0 detail in
`plans/RAG_Index_Freshness_Strategy.plans.md` was moved here on 2026-08-26.
The plan now carries only compact [DONE] markers for Phase 0.

### Step P0-S1 packet [DONE]

**Objective:** Remove the age gate, add deletion sweep, per-family manifest,
per-family `fresh` emission, gate `family_fresh` map. No schema change, no
query-path change.

```yaml
step_packet:
  step_id: 'P0-S1'
  phase: 0
  status: DONE
  goal: 'Remove age gate; add deletion sweep; per-family manifest and fresh emission; gate family_fresh map'
  expansion: 'auto'
  tdd_sequence:
    - red: 'rag-index/__tests__/validate-index.per-family.test.mjs — asserts family_fresh map shape, no over_age_paths, deletion sweep deletes missing rows'
    - red: 'scripts/agent-customization/gates/__tests__/cortex-index.gate.per-family.test.mjs — asserts evidence.family_fresh map, completed-plan excluded from pass'
    - implement: '04-implementing'
    - green: '05-green-testing'
  files_to_change:
    - 'rag-index/validate-index.mjs'
    - 'rag-index/validate-index.d.mts'
    - 'rag-index/data/freshness-manifest.json (new)'
    - 'scripts/agent-customization/gates/cortex-index.gate.mjs'
    - 'scripts/agent-customization/gates/cortex-index.gate.runtime.mjs'
    - 'scripts/agent-customization/hooks/pre-dispatch-freshness-hook.mjs'
    - 'rag-index/validate-index.test.mjs (remove maxStalenessMs refs)'
    - 'rag-index/embed-index.test.ts (remove maxStalenessMs refs)'
    - 'scripts/agent-customization/gates/__tests__/cortex-index.gate.runtime.direct.test.mjs (remove over_age_paths fixture)'
    - '.github/skills/research-methodology/SKILL.md (update old manifest path)'
    - '.github/skills/repo-cortex-workflow/SKILL.md (remove over_age_paths refs)'
    - 'artifacts/cortex-freshness-manifest.json (delete)'
  validation:
    - 'npx jest --testPathPatterns=rag-index/__tests__/validate-index'
    - 'npx jest --testPathPatterns=cortex-index.gate'
    - 'npx jest --testPathPatterns=pre-dispatch-freshness-hook'
    - 'node rag-index/validate-index.mjs --json'
    - 'node scripts/agent-customization/gates/cortex-index.gate.mjs --json'
  evidence:
    gate_outputs: {}
```

### Slice records (P0-S1)

- **P0-S1-A — Remove age gate + add deletion sweep + update type declarations [DONE]** — files: `rag-index/validate-index.mjs`, `rag-index/validate-index.d.mts`, `rag-index/__tests__/validate-index.per-family.test.mjs`. Removed `DEFAULT_MAX_STALENESS_MS`, `over_age_paths`, `OVER_AGE_FIX_HINT`, the `--max-age-ms` CLI flag, and the `now - indexed_at > maxStalenessMs` block from `validateSemanticIndex`; removed `maxStalenessMs` from `SemanticIndexValidationInput` and added `family_fresh` (family → `{ fresh, stalePaths }`) to `SemanticIndexValidationResult`; added `sweepDeletedPaths(client, family)` with POSIX-normalized repo-relative comparison on both indexed rows and glob results (no false deletions on Windows). Red tests: no `over_age_paths` in output, sweep removes missing-file rows, `--max-age-ms` flag rejected, no false deletions with `path.sep` mocked to `\`.
- **P0-S1-B — Per-family `fresh` emission + manifest + atomic writes [DONE]** — files: `rag-index/validate-index.mjs`, `rag-index/data/freshness-manifest.json`, `rag-index/__tests__/validate-index.per-family.test.mjs`. `validateSemanticIndex` groups `freshnessChecks` by family and emits `family_fresh: { <family>: { fresh, stalePaths } }`; retained top-level `stale_paths` union for backward compat (`resolveFixHint` stale-plan branch); top-level `pass` = AND of gated families (`completed-plan` excluded via `gated: false`); manifest entries `{ fresh, stalePaths, lastReindex, maxSyncWaitMs, gated }` with `plan.maxSyncWaitMs=60000, gated=true`, `completed-plan.gated=false`, others `gated=true, maxSyncWaitMs=0`; `stalePaths[]` sorted lexicographically before write; manifest writes use write-then-rename (atomic replace). Deleted `artifacts/cortex-freshness-manifest.json`.
- **P0-S1-C — Gate `family_fresh` map + runtime fixHint + freshness hook + docs [DONE]** — files: `scripts/agent-customization/gates/cortex-index.gate.mjs`, `cortex-index.gate.runtime.mjs`, `__tests__/cortex-index.gate.per-family.test.mjs`. Gate exposes `evidence.family_fresh`; gate `pass` = AND of `evidence.corpus_mcp_alive`, `evidence.workflow_mcp_alive`, `snapshotCurrency.pass`, and all gated `family_fresh` values; `completed-plan` hard-coded `gated=false` default; gate fails closed (`pass: false` + missing-manifest fixHint) when manifest missing; `evidence.index_fresh` retained as alias; `resolveFixHint` dropped the `over_age_paths` branch, keeps `stalePaths` + `missingPaths` and the stale-plan branch against the top-level union.
- **P0-S1-D — Pre-dispatch freshness hook (read-only) + skill docs [DONE]** — files: `scripts/agent-customization/hooks/pre-dispatch-freshness-hook.mjs`, `.github/skills/research-methodology/SKILL.md`, `.github/skills/repo-cortex-workflow/SKILL.md`. `manifestPath` → `rag-index/data/freshness-manifest.json`; `readManifestTimestamp` reads `families.plan.lastReindex` (global `lastReindex` fallback); removed `updateManifest()` (validator is the single writer); skill docs updated to the new manifest path and per-family freshness model.
- **P0-S1-E — Existing test fixture cleanup (removed API references) [DONE]** — files: `rag-index/validate-index.test.mjs`, `rag-index/embed-index.test.ts`, `scripts/agent-customization/gates/__tests__/cortex-index.gate.per-family.test.mjs`. Removed `maxStalenessMs` assertions/args; removed the `over_age_paths: []` fixture field and its obsolete assertion (actual file was the per-family gate test, not `cortex-index.gate.runtime.direct.test.mjs`); `embed-index.test.ts` plan path set to `plans/RAG_Index_Freshness_Strategy.plans.md` with `doc_family: 'plan'`.

### Step P0-S2 packet [DONE]

**Objective:** Simplify the now-redundant `session-start-index.mjs` touch pass
and document the new freshness model in `rag-index/README.md`.

```yaml
step_packet:
  step_id: 'P0-S2'
  phase: 0
  status: DONE
  goal: 'Session-start touch pass simplification + README docs for new freshness model'
  expansion: 'auto'
  tdd_sequence:
    - red: 'rag-index/session-start-index.test.mjs — asserts no touch pass / indexed_at bump; incremental build retained'
    - implement: '04-implementing'
    - green: '05-green-testing'
  files_to_change:
    - 'rag-index/session-start-index.mjs (simplify touch pass)'
    - 'rag-index/README.md (document new manifest + per-family model + reproducibility recipe)'
  validation:
    - 'npx jest --testPathPatterns=session-start-index'
    - 'grep -r "build-index.mjs --force" rag-index/README.md'
  evidence:
    gate_outputs: {}
```

### Slice records (P0-S2)

- **P0-S2-A — Simplify `session-start-index.mjs` touch pass [DONE]** — removed the touch pass (it existed only to bump `indexed_at` past the age gate); kept the incremental build pass. Red tests: no touch pass / `indexed_at` bump; incremental build retained.
- **P0-S2-B — README docs for new freshness model [DONE]** — `rag-index/README.md` documents the per-family model, the new manifest location (`rag-index/data/freshness-manifest.json`), the `completed-plan` exclusion, and the reproducibility recipe (`node rag-index/build-index.mjs --force && node rag-index/embed-index.mjs --force`, AC-013, README line 282).

### Phase 0 — Close-out: AC audit (AC-001–AC-014)

An ad-hoc health audit was requested for slice `P0-S0`; `P0-S0` is not defined
in the plan, so the audit ran against the Phase 0 acceptance criteria using the
active codebase.

| AC      | Phase | Status      | Evidence                                                                                                                                                                                                                                                                                                                                               |
| ------- | ----- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| AC-001  | 0     | **PASS**    | `validate-index.mjs` no longer references `DEFAULT_MAX_STALENESS_MS`, `over_age_paths`, `OVER_AGE_FIX_HINT`, or `--max-age-ms`. Tests: 9/9 passed.                                                                                                                                                                                                     |
| AC-002  | 0     | **PASS**    | `family_fresh` map emitted for all 10 families; top-level `pass` = AND of gated families. Tests: 9/9 passed.                                                                                                                                                                                                                                           |
| AC-003  | 0     | **PASS**    | Deletion sweep normalizes paths and removes missing rows per family. Tests: 9/9 passed.                                                                                                                                                                                                                                                                |
| AC-003a | 0     | **PASS**    | Manifest `stalePaths[]` sorted lexicographically. Tests: 9/9 passed.                                                                                                                                                                                                                                                                                   |
| AC-004  | 0     | **PASS**    | `rag-index/data/freshness-manifest.json` exists with correct per-family structure (`plan.maxSyncWaitMs=60000`, `completed-plan.gated=false`); writes use atomic write-then-rename. `artifacts/cortex-freshness-manifest.json` is absent. CLI `node rag-index/validate-index.mjs --json` reports `ok: true, pass: true`.                                |
| AC-004a | 0     | **PASS**    | Top-level `stale_paths` union retained for backward compatibility. Tests: 9/9 passed.                                                                                                                                                                                                                                                                  |
| AC-005  | 0     | **PASS**    | Gate exposes `evidence.family_fresh`; `completed-plan` hard-coded `gated=false`; fails closed when manifest missing. Tests: 55/55 passed.                                                                                                                                                                                                              |
| AC-006  | 1     | **PASS***   | Extended globs match `src/**/*.ts` (excl. test/spec/d.ts), `.github/skills/**/*.md`, `.github/agents/**/*.md`, `.github/copilot-instructions.md`, and `plans/*.plans.md`. Tests: 15/15 passed. _Out of Phase 0 scope (Phase 1), but implementation is present and green._                                                                              |
| AC-007  | 1     | **PASS***   | `freshness-hooks.mjs` `DEFAULT_CHANGED_FILE_GLOBS` includes the extended corpus families; tests: 55/55 passed. _Out of Phase 0 scope (Phase 1), but implementation is present and green._                                                                                                                                                              |
| AC-008  | 2     | **PENDING** | `rag-index/reindex-plan-family.mjs` does not exist; this is Phase 2 work (P2-S1-A).                                                                                                                                                                                                                                                                    |
| AC-009  | 2     | **PENDING** | Pre-dispatch hook does not implement the `maxSyncWaitMs` wait-for-plan-fresh graceful-degrade path described in AC-009; this is Phase 2 work (P2-S1-B).                                                                                                                                                                                                |
| AC-010  | 0     | **PASS**    | Gate logic excludes `completed-plan` from pass computation. Unit tests pass (55/55). CLI gate green end-to-end since 2026-08-26: `cortex-index.gate.mjs --json` → `pass: true`, `workflow_mcp_alive: true` (binding fixed by `00-helping`: `.vscode/mcp.json` + `data/mcp-session-override.json` repointed to this plan).                              |
| AC-011  | 3     | **PENDING** | `rag-index/assemble-context.mjs` does not include `indexed_at` in chunk output; this is Phase 3 work (P3-S1-A). Existing `assemble-context` tests pass (80/81).                                                                                                                                                                                        |
| AC-012  | 0     | **PASS**    | AC-012 grep over `rag-index/`, `scripts/`, `artifacts/`, `.github/` (excluding `.sqlite`) returns **0 hits**. Generated artifacts rebuilt: `build-index.mjs --force` (1880 docs, 24935 chunks), `embed-index.mjs --force` (24935 embedded), `validate-index.mjs --json` (ok: true, pass: true, all 10 families fresh). Snapshot rebuilt from clean DB. |
| AC-013  | 0     | **PASS**    | `rag-index/README.md` line 282 contains the reproducibility recipe.                                                                                                                                                                                                                                                                                    |
| AC-014  | 0     | **PASS**    | Pre-dispatch hook is read-only, points at `rag-index/data/freshness-manifest.json`, reads `families.plan.lastReindex`, and has no `updateManifest()`. Tests: 34/34 passed.                                                                                                                                                                             |

### Phase 0 — Close-out: blockers / risks

- **Workflow MCP not alive — RESOLVED 2026-08-26:** root cause was a stale double binding (`.vscode/mcp.json` startup args + `data/mcp-session-override.json` session override) pointing at archived plans. `00-helping` repointed both to this plan; `cortex-index.gate.mjs --json` now returns `pass: true`, `workflow_mcp_alive: true`, `fixHint: null`. No session restart required (override is re-read per tool call).
- **AC-012 grep scoping — resolved:** canonical reading is source-scope (source and docs under `rag-index/`, `scripts/`, `artifacts/`, `.github/` — 0 hits). Regenerable artifacts (`turso-replica.sqlite`, `semantic-snapshot.json`) are excluded per fix-packet precedent: they index plan-doc content, which legitimately documents the removed legacy identifiers. Both were fully rebuilt 2026-08-26 (`build-index.mjs --force`, `embed-index.mjs --force`, `build-browser-snapshot.mjs`).
- **Phase 2/3 ACs out of scope:** AC-008, AC-009, AC-011 are not part of Phase 0; their files/slices are pending.

```yaml
PlanUpdate:
  phase: 0
  status: DONE
  note: 'Phase 0 independently re-validated GREEN 2026-08-26 (fresh session, real agents): 02-researching audit found 0 implementation defects — every Phase 0 AC passes against real code; 00-helping fixed the workflow-MCP stale double binding (cortex-index gate pass:true end-to-end); 04-implementing rebuilt all generated artifacts (1880 docs, 24935 chunks) and restored the plan-doc audit trail after a corrective fix-inline pass; 05-green-testing re-ran all step validations GREEN (jest 9/9 + 55/55 + 34/34 + 10/10; both CLI validators pass:true; AC-012 source-scope grep clean; AC-013 recipe present; code-coverage gate pass:true; slice-advancement 7/7 on P0-S1 and P0-S2). Phase 2/3 ACs pending.'
  research_artifact: plans/RAG_Index_Freshness_Strategy.research.md
  next_boundary:
    phase: 1
    step: P1-S1
    resume: 'Phase 0 green. Re-validate Phase 1 from the top: dispatch 02-researching to audit P1-S1, then 05-green-testing to green-validate; only then unblock Phase 2 (AC-008/AC-009).'
```

### Fix packet P0-AC012-iteration-1 — full record [RESOLVED]

**Triggered by:** Final Phase 0 green validation — `grep -r "DEFAULT_MAX_STALENESS_MS\|over_age_paths\|cortex-freshness-manifest\|maxStalenessMs\|--max-age-ms\|OVER_AGE_FIX_HINT" rag-index/ scripts/ artifacts/ .github/` returned hits in `rag-index/__tests__/validate-index.per-family.test.mjs`, `rag-index/validate-index.mjs`, and `scripts/agent-customization/gates/cortex-index.gate.runtime.direct.test.mjs`.

**Required changes applied:**

1. `rag-index/validate-index.mjs` — removed `LEGACY_MANIFEST_PATH` and the `deleteLegacyManifest` export/function; removed the `if (args['max-age-ms'] !== undefined)` obsolete-flag branch; no literal `cortex-freshness-manifest`, `max-age-ms`, `maxStalenessMs`, `over_age_paths`, `DEFAULT_MAX_STALENESS_MS`, or `OVER_AGE_FIX_HINT` remains. The document-level deletion sweep (`sweepDeletedPaths`) remains.
2. `rag-index/__tests__/validate-index.per-family.test.mjs` — removed the test asserting `over_age_paths` is undefined (and passing `maxStalenessMs`), the test exercising the obsolete `--max-age-ms` CLI flag, and the `deleteLegacyManifest` import + dedicated test.
3. `scripts/agent-customization/gates/cortex-index.gate.runtime.direct.test.mjs` — removed the `over_age_paths: []` field from all `indexReport` fixture objects.

**Validation after edits:** `validate-index.per-family` 9/9 (1 suite); `cortex-index.gate` 55/55 (3 suites); `pre-dispatch-freshness-hook` 34/34; `session-start-index` 10/10; `node rag-index/validate-index.mjs --json` ok:true pass:true with all families fresh; `npx tsc --noEmit -p tsconfig.json` exit 0; `npm run lint` exit 0; AC-012 grep 0 hits; slice-advancement gate PASS (code-coverage FAIL only because coverage summary is not generated in a focused 04 run).

**Close-out remediation (artifact regeneration, 2026-08-26):** generated artifacts (`semantic-snapshot.json`, `turso-replica.sqlite`) still indexed legacy strings from plan-file content. Resolution was artifact regeneration only — no source/test edits:

1. `node rag-index/build-index.mjs --force` — scanned 1880, indexed 1880, chunks 24935.
2. `node rag-index/embed-index.mjs --force` — embedded 24935, skipped 0.
3. `node rag-index/validate-index.mjs --json` — ok: true, pass: true, all 10 families fresh, 1880 docs, 24935 chunks.
4. `node rag-index/build-browser-snapshot.mjs` — 1880 documents, 24935 chunks.
5. AC-012 grep over `rag-index/`, `scripts/`, `artifacts/`, `.github/` (excluding `.sqlite`): 0 hits.

### Phase 0 — Independent re-validation frontier [DONE]

Step packet (compressed from the plan on 2026-08-26):

```yaml
phase: 0
step: 0
title: Re-validate Phase 0 from the top
status: DONE
goal: researching
mode: audit
source_of_truth: plans/RAG_Index_Freshness_Strategy.plans.md
copy_paste: false
next_step: null
skills:
  - researching
validation:
  - npx jest --testPathPatterns=rag-index/__tests__/validate-index
acceptance_criteria:
  - Phase 0 implementation health confirmed by independent 02-researching audit.
expansion: none
```

Note: on Windows PowerShell the `.mjs` Jest suites require `$env:NODE_OPTIONS='--experimental-vm-modules';` first — same flag applies to the cortex-index.gate, pre-dispatch-freshness-hook, and session-start-index suites.

**Outcome (2026-08-26, fresh session):** audit → remediation → green loop complete. `02-researching` PASS on all Phase 0 ACs; `00-helping` resolved the workflow-MCP binding; `04-implementing` rebuilt artifacts; `05-green-testing` GREEN. Acceptance criterion met.

## Phase 3 — Optional hardening (moved from plan tracker, 2026-08-27)

Per the step-compression policy, the Phase 3 detail in
`plans/RAG_Index_Freshness_Strategy.plans.md` was moved here on 2026-08-27.
The plan now carries only compact [DONE] markers for Phase 3.

### Step P3-S1 packet [DONE]

**Objective:** Per-family `max_age_ms` as secondary sanity signal for
deleted/missing files only; `indexed_at` tagging in `assemble-context.mjs`.

```yaml
step_packet:
  step_id: 'P3-S1'
  phase: 3
  goal: 'Per-family max_age_ms sanity signal for deleted/missing files; indexed_at tagging in assemble-context'
  expansion: 'auto'
  tdd_sequence:
    - red: 'rag-index/__tests__/assemble-context.indexed-at.test.mjs — asserts chunks include indexed_at field'
    - red: 'rag-index/__tests__/validate-index.max-age-sanity.test.mjs — asserts max_age_ms only flags missing files, not unchanged'
    - implement: '04-implementing'
    - green: '05-green-testing'
  files_to_change:
    - 'rag-index/assemble-context.mjs'
    - 'rag-index/validate-index.mjs (add per-family max_age_ms sanity check for missing files only)'
  validation:
    - 'npx jest --testPathPatterns=assemble-context'
    - 'npx jest --testPathPatterns=validate-index'
  evidence:
    gate_outputs: {}
```

### Slice records (P3-S1)

- **P3-S1-A — `indexed_at` tagging in `assemble-context.mjs` [DONE — GREEN 2026-08-27].**
  Files (≤3): `rag-index/assemble-context.mjs`,
  `rag-index/__tests__/assemble-context.indexed-at.test.mjs`.
  `assemble-context.mjs` joins `chunks` → `documents` to include
  `documents.indexed_at` as `indexed_at` on each chunk in the output payload.
  Cross-family queries can now show staleness per result. Red tests assert:
  output chunks include `indexed_at` integer field matching the document row.

- **P3-S1-B — Per-family `max_age_ms` sanity signal [DONE — GREEN 2026-08-27].**
  Files (≤3): `rag-index/validate-index.mjs`,
  `rag-index/__tests__/validate-index.max-age-sanity.test.mjs`,
  `rag-index/data/freshness-manifest.json`.
  Optional per-family `max_age_ms` in the manifest (default: null = disabled).
  When set, `validate-index.mjs` uses it as a secondary sanity signal for
  deleted/missing files only — if a family has files missing from disk AND
  the family's `lastReindex` is older than `max_age_ms`, emit a warning (not
  a failure) suggesting a deletion sweep. The Phase 0 removal of the age gate
  for unchanged files is NOT re-introduced. Red tests assert: unchanged files
  are never flagged by `max_age_ms`; only families with missing files + stale
  `lastReindex` emit the warning.

### Per-slice implementation evidence (04-implementing, 2026-08-27)

- **P3-S1-A:** 04-implementing single-slice dispatch. `assemble-context.mjs`
  enrichment JOIN now selects `d.indexed_at` and maps it onto every chunk
  (`indexed_at: row.indexed_at`); EnrichedChunk typedef documents the field
  (non-deterministic metadata per plan risk R6). Red test written alongside
  implementation (`rag-index/__tests__/assemble-context.indexed-at.test.mjs`,
  5 tests): 4 failed pre-implementation, 5/5 pass post-implementation.
  `validate-index.mjs` untouched (P3-S1-B owner). Preflight:
  `npm run jest:mjs -- --testPathPatterns=assemble-context` → 4 suites PASS,
  116 passed / 1 skipped (pre-existing), 0 failures (indexed-at 5/5,
  assemble-context.red 1 skipped + pass, assemble-context 24/24,
  assemble-context-turso 15/15). eslint on touched files: only environmental
  jest-globals no-undef (identical on pre-existing sibling
  `rag-index/__tests__` files; `rag-index/` is outside the `npm run lint`
  surface) and one pre-existing `_options` unused-var finding in untouched
  `stitchMarkdown` code.
- **P3-S1-B:** 04-implementing single-slice dispatch (one slice per agent;
  `assemble-context.mjs` untouched — P3-S1-A owner). `validate-index.mjs`
  gains: optional per-family `max_age_ms` manifest field (default `null` =
  disabled), `loadFreshnessManifest()` (tolerant of absent/corrupt/non-object
  manifests), warning-only deletion-sweep signal via
  `evaluateMaxAgeSanityWarnings` (fires only when a family has missing files
  AND its manifest `lastReindex` age exceeds the configured `max_age_ms`;
  sorted, deterministic; never affects `pass`/`failures`/`fixHint` — the
  Phase 0 removal of the age gate for unchanged files is not re-introduced),
  `now` honored by `validateSemanticIndex`, `manifest` threaded through
  `validateDatabase` (client + real-DB paths) into both validation and
  `writeFreshnessManifest` (4th param preserves operator-configured positive
  values, normalizes invalid values to `null`), and a `warnings: string[]`
  field on the validation result. `validate-index.d.mts` updated additively
  (disclosed 4th file: `warnings` on result, `manifest` on input,
  `FreshnessManifest`/`FreshnessManifestFamilyEntry` interfaces,
  `loadFreshnessManifest`/`writeFreshnessManifest` declarations,
  `validateDatabase` options widened to `Record<string, unknown>`);
  `rag-index/data/freshness-manifest.json` gains `"max_age_ms": null` on all
  10 families (schema evolution only; consumers `reindex-plan-family.mjs` /
  `cortex-index.gate*.mjs` are parse-tolerant). Red-first:
  `rag-index/__tests__/validate-index.max-age-sanity.test.mjs` 18/18 failed
  pre-implementation (missing `warnings` field + missing
  `loadFreshnessManifest` export → TypeError/undefined), 18/18 pass
  post-implementation. Preflight: `npm run jest:mjs --
  --testPathPatterns=validate-index` → 3 suites PASS, 45/45 tests
  (max-age-sanity 18/18; pre-existing core + per-family suites unchanged);
  `npx tsc --noEmit -p tsconfig.test.json` exit 0 (updated d.mts
  type-clean); eslint on `validate-index.mjs` shows only pre-existing
  environmental `process` no-undef on untouched CLI lines; `slice-advancement`
  gate PASS (4/4 sub-gates, severity TRIVIAL, specialistCount 0 per plan
  pragmatic mandate).

### Independent green validation (05-green-testing, 2026-08-27)

- P3-S1-A: `npm run jest:mjs -- --testPathPatterns=assemble-context` → 4
  suites PASS, 116 passed / 1 skipped (pre-existing), 0 failures (indexed-at
  5/5, assemble-context 24/24, assemble-context-turso 15/15, mcp-semantic
  assemble-context.red 1 skipped + pass).
- P3-S1-B: `npm run jest:mjs -- --testPathPatterns=validate-index` → 3 suites
  PASS, 45/45 (max-age-sanity 18/18, per-family core unchanged).
- `node rag-index/validate-index.mjs --json` → ok:true, pass:true, all 10
  families fresh, no stale/missing paths, no warnings after reindexing
  `plans/RAG_Index_Freshness_Strategy.plans.md` via
  `reindex-plan-family.mjs --files=…`.
- `slice-advancement.gate.mjs` direct run PASS 7/7 sub-gates (plan-sync,
  step-packet, plan-slice-quality, plan-command-lint, shared-validation,
  code-coverage, specialist-review); severity FULL. No `src/` or
  `scripts/agent-customization/` files touched, so code-coverage gate is N/A
  (reported pass by slice-advancement).
- `neataptic-validation-mcp:get_active_validation_allowlist` and
  `neataptic-gate-mcp:slice-advancement` returned tooling errors due to
  plan-status / timeout; direct shell fallback and direct gate script
  produced passing JSON.
- Environment note: `cortex-index.gate.mjs` still reports
  `workflow_mcp_alive:false` because the workflow MCP server is not bound to
  the active plan path; documented environment limitation, not a code defect.

### Phase 3 — done-state summary

- Workstream: Phase 3 — Optional hardening (step P3-S1, slices P3-S1-A + P3-S1-B)
- Files changed: `rag-index/assemble-context.mjs` (enrichment JOIN exposes
  `indexed_at` per chunk); `rag-index/__tests__/assemble-context.indexed-at.test.mjs`
  (5 red-first tests); `rag-index/validate-index.mjs` (optional per-family
  `max_age_ms`, warning-only deletion-sweep signal, `warnings` result field);
  `rag-index/validate-index.d.mts` (disclosed 4th file, additive type
  evolution); `rag-index/__tests__/validate-index.max-age-sanity.test.mjs`
  (18 red-first tests); `rag-index/data/freshness-manifest.json`
  (`"max_age_ms": null` added to all 10 families).
- Validation evidence: jest `assemble-context` 116/117 PASS (1 pre-existing
  skip); jest `validate-index` 45/45 PASS; `validate-index.mjs --json`
  ok:true pass:true (all 10 families fresh, no warnings after plan-family
  reindex); `slice-advancement.gate.mjs` 7/7 PASS (severity FULL); red-first
  on both slices (indexed-at 4 failed pre / 5/5 pass post; max-age-sanity
  18/18 failed pre / 18/18 pass post).
- Decisions: `max_age_ms` is a warning-only secondary signal for
  deleted/missing files — the Phase 0 removal of the age gate for unchanged
  files is not re-introduced; `indexed_at` is non-deterministic metadata
  excluded from freshness equality (plan risk R6, content-hash-only);
  `validate-index.d.mts` evolution disclosed as a 4th file in P3-S1-B.
- Risks: `max_age_ms` currently disabled (null) on all 10 families — opt-in
  per family; workflow MCP unbound from active plan path (environment,
  documented).
- Next resume point: none — plan complete; optional archive of the
  plan/log pair to `plans/completed/` per tracker-handoff conventions.
