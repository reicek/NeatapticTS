# RAG Index Freshness Strategy — Phase 0 Health Audit Research

## Question

The user asked to "execute slice P0-S0" and report PASS/FAIL for acceptance criteria AC-001 through AC-014 on `plans/RAG_Index_Freshness_Strategy.plans.md`. Slice `P0-S0` is not defined in the plan; Phase 0 implementation appears complete (P0-S1 and P0-S2 are marked `[DONE]`). This artifact records the evidence and decisions from the ad-hoc health audit.

## Evidence

1. **Slice identity** — `neataptic-workflow-mcp-get_slice_context` returned `notFound: true` for `slice_id: P0-S0`. The plan tracker only defines P0-S1 and P0-S2, both `[DONE]`.

2. **Test suite results** (all run with `NODE_OPTIONS="--experimental-vm-modules"` because `.mjs` tests need native ESM):
   - `rag-index/__tests__/validate-index.per-family.test.mjs` — 9/9 passed.
   - `scripts/agent-customization/gates/__tests__/cortex-index.gate.per-family.test.mjs` (and related gate suites) — 55/55 passed across 3 suites.
   - `scripts/agent-customization/hooks/pre-dispatch-freshness-hook.test.mjs` — 34/34 passed.
   - `rag-index/__tests__/auto-reindex.extended-glob.test.mjs` + `rag-index/auto-reindex.test.mjs` — 15/15 passed.
   - `rag-index/__tests__/freshness-hooks.extended-glob.test.mjs` + `rag-index/freshness-hooks/freshness-hooks.test.mjs` — 55/55 passed.
   - `rag-index/__tests__/assemble-context*.test.mjs` — 80/81 passed (1 pending).

3. **CLI validators**:
   - `node rag-index/validate-index.mjs --json` — reports `ok: true, pass: true` with all 10 families fresh and no `stale_paths`/`missing_paths`.
   - `node scripts/agent-customization/gates/cortex-index.gate.mjs --json` — returns `pass: false` because `workflow_mcp_alive: false`; `family_fresh` and `index_fresh` are all true. The failure is environmental, not a freshness defect.

4. **Source inspections**:
   - `rag-index/validate-index.mjs` no longer references age-gate terms; has `sweepDeletedPaths`, `buildFamilyFresh`, atomic `writeFreshnessManifest`.
   - `rag-index/validate-index.d.mts` has `family_fresh` in result; no `maxStalenessMs` in input.
   - `rag-index/data/freshness-manifest.json` has per-family structure with `plan.maxSyncWaitMs=60000` and `completed-plan.gated=false`.
   - `scripts/agent-customization/gates/cortex-index.gate.mjs` exposes `evidence.family_fresh`; hard-codes `completed-plan: gated=false`; fails closed when manifest missing.
   - `scripts/agent-customization/hooks/pre-dispatch-freshness-hook.mjs` is read-only, reads `families.plan.lastReindex`, no `updateManifest()`.
   - `rag-index/auto-reindex.mjs` and `rag-index/freshness-hooks/freshness-hooks.mjs` include extended globs for source, skills, agents, copilot-instructions, and plans.
   - `rag-index/README.md` line 282 documents the reproducibility recipe.

5. **Missing / out-of-phase items**:
   - `rag-index/reindex-plan-family.mjs` does not exist. It is scoped to Phase 2 (P2-S1-A/B), so its absence is expected for a Phase 0 audit.
   - `rag-index/assemble-context.mjs` does not include `indexed_at` in output. It is scoped to Phase 3 (P3-S1-A).

6. **Legacy-string search**:
   - Source-only grep (excluding `.sqlite` and generated `.json` snapshots) for `DEFAULT_MAX_STALENESS_MS`, `over_age_paths`, `cortex-freshness-manifest`, `maxStalenessMs`, `--max-age-ms`, `OVER_AGE_FIX_HINT` returned no hits.
   - Generated `rag-index/snapshots/semantic-snapshot.json` and `rag-index/data/turso-replica.sqlite` still contain indexed historical references to these terms. They are generated/indexed artifacts, not source code.

## Decision

- AC-001–AC-005, AC-010, AC-012–AC-014 are **PASS** for Phase 0.
- AC-006 and AC-007 are **PASS** in the current code but belong to Phase 1; they should be recorded as green but out-of-phase for a strict Phase 0 audit.
- AC-008 and AC-009 are **PENDING** (Phase 2); the required `reindex-plan-family.mjs` file is missing.
- AC-011 is **PENDING** (Phase 3); `assemble-context.mjs` lacks `indexed_at` tagging.
- The `cortex-index` gate CLI failure is environmental (`workflow_mcp_alive: false`), not a Phase 0 implementation defect.
- The literal AC-012 grep is not clean because generated snapshot/DB artifacts still index legacy terms. A full reindex (`build-index.mjs --force && embed-index.mjs --force`) is needed to make the literal grep pass.

## Risks

- **Workflow MCP binding:** The gate CLI cannot be fully green until the workflow MCP server is alive/bound. This masks end-to-end verification.
- **Generated artifact staleness:** Snapshot and SQLite DB still embed pre-Phase-0 terminology. If AC-012 is enforced literally over all files, a full reindex is required.
- **Scope ambiguity:** AC-008/AC-009/AC-011 are part of later phases. If the caller expected them to be implemented in Phase 0, the plan phase mapping must be revisited.

## Phase 1 Slice P1-S1 Re-validation Audit

### Question
Do the current repository implementations of Phase 1 slice P1-S1 satisfy acceptance criteria AC-006 (`auto-reindex.mjs` post-commit glob extension) and AC-007 (`freshness-hooks.mjs` watcher glob extension) when audited against the real code?

### Evidence

- **AC-006 source inspection (`rag-index/auto-reindex.mjs`)**:
  - `REINDEXABLE_FAMILIES` matches `plans/*.plans.md`, `src/**/*.ts` excluding `.test.ts`/`.spec.ts`/`.d.ts`, `.github/skills/**/*.md`, `.github/agents/**/*.md`, and `.github/copilot-instructions.md`.
  - `resolveStalePlanFixHint` remains plan-only via `isPlanFile`.
  - Identifiers renamed to `reindexChangedFiles`/`detectChangedFiles`; log path is `auto-reindex.log`.
  - Post-commit hook comment describes general corpus reindex.
  - Test: `rag-index/__tests__/auto-reindex.extended-glob.test.mjs` — 3/3 passed under `rag-index-mjs` project with `--experimental-vm-modules`.

- **AC-007 source inspection (`rag-index/freshness-hooks/freshness-hooks.mjs`)**:
  - `DEFAULT_CHANGED_FILE_GLOBS` includes `src/**/*.ts`, `scripts/**/*.mjs`, `plans/**/*.md`, `.github/skills/**/SKILL.md`, `.github/agents/*.agent.md`, and `.github/copilot-instructions.md`.
  - `FAMILY_RULES` includes `skill`, `agent`, and `copilot-instructions` families.
  - `DEFAULT_IGNORE_GLOBS` filters `.test.ts`, `.spec.ts`, `.d.ts` before the incremental builder.
  - Test: `rag-index/__tests__/freshness-hooks.extended-glob.test.mjs` — 2/2 passed under `rag-index-mjs` project with `--experimental-vm-modules`.
  - **Discrepancy**: `auto-reindex.mjs` matches *all* `.md` files under `.github/skills/` and `.github/agents/` (including reference/asset files and `.github/agents/README.md`). `freshness-hooks.mjs` only watches `.github/skills/**/SKILL.md` and `.github/agents/*.agent.md`. As of audit date, 23 skill `.md` files and 1 agent `.md` file are covered by post-commit but not by the watcher hook.

- **Index health**: `node rag-index/validate-index.mjs --json` reports all 10 families fresh with no stale or missing paths (`ok: true, pass: true`).

- **Plan/RAG issue**: `neataptic-workflow-mcp-get_slice_context` for `slice_id: P1-S1` returned `notFound: true` because the plan's step packet uses `step_id: 'P1-S1'` rather than `slice_id: 'P1-S1'`. Direct plan document read was used as fallback.

### Decision

- **AC-006: PASS** — implementation fully matches the acceptance criterion and slice contract.
- **AC-007: PARTIAL / CONDITIONAL FAIL** — required new globs are present and tests pass, but the watcher glob set is narrower than the corpus set reindexed by `auto-reindex.mjs`, leaving 24 `.md` files uncovered by the post-write watcher hook.

### Risks

- **Watcher / post-commit inconsistency**: 24 corpus `.md` files are reindexed on commit but not on editor/MCP post-write, risking silent drift.
- **Silent index drift**: Gap is invisible until one of the uncovered files is edited and ignored by the watcher.
- **Family inference mismatch**: Files reaching the builder through non-watcher paths receive family `unknown`.
- **Plan format issue**: `get_slice_context` failure blocks RAG-based dispatch until the step packet is normalized to use `slice_id`.
