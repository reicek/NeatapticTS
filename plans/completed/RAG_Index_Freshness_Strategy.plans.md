# RAG Index Freshness Strategy

**Status:** [DONE]

> Incident/recovery notes, completed Phase 0/1/2/3 implementation details, the
> Phase 0 close-out AC audit, fix packet P0-AC012-iteration-1, and the Phase 0
> re-validation step packet have been compressed into
> `plans/RAG_Index_Freshness_Strategy.logs.md` (§ "Phase 0 — Full detail
> (moved from plan tracker, 2026-08-26)"; Phase 3 compressed 2026-08-27, §
> "Phase 3 — Optional hardening (moved from plan tracker, 2026-08-27)").
> This tracker now carries only compact [DONE] markers, plan-level reference
> material, and the handoff query.

## Purpose

Replace the content-blind 24h age gate in `rag-index/validate-index.mjs` with a
per-family, content-hash-only freshness model on the existing single Turso DB.
The current `DEFAULT_MAX_STALENESS_MS = 24h` flag unchanged files as stale purely
by age, triggering full background reindexes (e.g., the 5.5-day-old index that
just fired a reindex). The fix introduces a per-family freshness manifest, a
deletion sweep that closes the gap the age gate was masking, per-family `fresh`
emission from the validator and gate, extended post-commit/watcher hooks, a
synchronous post-save plan hook with 60s budget + graceful degrade, and optional
`indexed_at` tagging for cross-family staleness visibility.

**This plan is planning only. No code is implemented here.**

## Scope

In scope:

- `rag-index/validate-index.mjs` — remove the 24h age gate (`over_age_paths`),
  add per-family deletion sweep, emit per-family `fresh` map instead of global
  `pass`. Retain a top-level `stale_paths` (union of all families' stale
  paths) for backward compatibility so `cortex-index.gate.runtime.mjs`
  `resolveFixHint` stale-plan branch continues to work.
- `rag-index/validate-index.d.mts` — remove `maxStalenessMs` from
  `SemanticIndexValidationInput`; add `family_fresh` to
  `SemanticIndexValidationResult`.
- `rag-index/freshness.mjs` — no change to `getFreshnessProof` / `isFreshDocument`
  (already content-hash-only); reused as-is.
- `rag-index/data/freshness-manifest.json` — new per-family manifest (moved from
  `artifacts/cortex-freshness-manifest.json`), one `fresh` boolean per family,
  `stalePaths[]`, `lastReindex`, `maxSyncWaitMs` for plan (60s), `gated: false`
  for `completed-plan`.
- `scripts/agent-customization/gates/cortex-index.gate.mjs` — expose
  `evidence.family_fresh` as a map so orchestrator can dispatch on
  `plan.fresh` independently; drop reliance on global `index_fresh`.
  Hard-code `completed-plan: gated=false` default so it does not depend on
  manifest presence. Gate fails closed (`pass: false`) when manifest is
  missing.
- `scripts/agent-customization/gates/cortex-index.gate.runtime.mjs` —
  `resolveFixHint()` removes the `over_age_paths` branch (no longer emitted
  by the validator); simplifies to `stalePaths` + `missingPaths` only.
- `scripts/agent-customization/hooks/pre-dispatch-freshness-hook.mjs` —
  update `manifestPath` to `rag-index/data/freshness-manifest.json`;
  adapt `readManifestTimestamp` to read `families.plan.lastReindex` (or
  global `lastReindex`); **remove `updateManifest()`** — the validator is
  the single writer (manifest write concurrency hazard). Hook becomes
  read-only.
- `rag-index/auto-reindex.mjs` — extend post-commit glob from `plans/*.plans.md`
  to include `src/**/*.ts` (excl. test/spec/d.ts), `.github/skills/**/*.md`,
  `.github/agents/**/*.md`, `.github/copilot-instructions.md`.
- `rag-index/freshness-hooks/freshness-hooks.mjs` — extend
  `DEFAULT_CHANGED_FILE_GLOBS` to match the extended post-commit glob set.
- `rag-index/reindex-plan-family.mjs` — new synchronous post-save plan hook:
  sync BM25 via `build-index.mjs --files=…`, background dense via
  `embed-index.mjs --files=…`; pre-dispatch freshness hook waits up to 60s for
  `families.plan.fresh`, then graceful-degrades with a warning.
- `rag-index/assemble-context.mjs` — tag chunks with `indexed_at` in output so
  cross-family queries can show staleness per result (Phase 3, optional).
- `rag-index/watch-plans.mjs` — no change required (already watches plans); the
  new `reindex-plan-family.mjs` is the synchronous hook it invokes.
- `rag-index/session-start-index.mjs` — touch pass becomes redundant once the
  age gate is removed; simplify or remove in Phase 0 cleanup.
- `rag-index/README.md`, `rag-index/schema-turso.sql` — document the new
  manifest location and per-family freshness model.
- `.github/skills/repo-cortex-workflow/SKILL.md`,
  `.github/skills/repo-cortex-embeddings/SKILL.md`,
  `.github/skills/research-methodology/SKILL.md` — update freshness docs
  (research-methodology references the old manifest path
  `artifacts/cortex-freshness-manifest.json`).
- `artifacts/cortex-freshness-manifest.json` — removed (replaced by
  `rag-index/data/freshness-manifest.json`).

Out of scope (do not modify):

- Physical sharding of the Turso DB (cross-family queries are the norm; single
  DB retained).
- `rag-index/freshness.mjs` content-hash algorithm (already correct:
  sha256 + mtime + size).
- Core NEAT algorithms, examples, benchmarks.
- Cloud/Turso deployment topology.
- Historical completed plans under `plans/completed/` and the immutable
  `.github/ai-learning/learning-log.jsonl`.

## Architecture decisions (approved)

1. **No physical sharding** — keep single Turso DB; use `doc_family` column +
   metadata filter (cross-family queries are the norm).
2. **Remove the 24h age gate** (Q3a) — staleness is content-hash-only
   (`sha256` + mtime + size from `freshness.mjs`).
3. **Add deletion sweep** — `validate-index.mjs` compares indexed paths per
   family vs glob of current files; deletes DB rows for missing files (closes
   the gap the age gate was masking).
4. **Per-family freshness manifest** at `rag-index/data/freshness-manifest.json`
   (moved from `artifacts/`, Q7) — one `fresh` boolean per family,
   `stalePaths[]`, `lastReindex`, `maxSyncWaitMs` for plan (60s, Q2),
   `gated: false` for `completed-plan`.
5. **`validate-index.mjs` emits per-family `fresh`** instead of global `pass`.
6. **`cortex-index.gate.mjs` exposes `evidence.family_fresh`** as a map so
   orchestrator can dispatch on `plan.fresh` independently.
7. **`completed-plan` excluded from default gate** (Q5 — immutable cold tier,
   manual `npm run index:archive` or CI only).
8. **Extend post-commit hook** (Q4) from `plans/*.plans.md` to include
   `src/**/*.ts` (excl. test/spec/d.ts), `.github/skills/**/*.md`,
   `.github/agents/**/*.md`, `.github/copilot-instructions.md`.
9. **Extend `freshness-hooks.mjs` watcher globs** to match.
10. **Synchronous post-save plan hook** (Q2, 60s budget) — new
    `reindex-plan-family.mjs`: sync BM25 via `build-index.mjs --files=…`,
    background dense via `embed-index.mjs --files=…`; pre-dispatch freshness
    hook waits up to 60s for `families.plan.fresh`, then graceful-degrades with
    a warning.
11. **Tag chunks with `indexed_at`** in `assemble-context.mjs` output so
    cross-family queries can show staleness per result.

## Mandates

**Pragmatic mode is authorized for this plan.** This is a focused infrastructure
change with a tight, well-understood file set. The following mandates apply:

- **Broad slices OK** — one dispatch per phase is acceptable; slices may span
  the full phase file set when the behavioral intent is cohesive.
- **Single-model mandate** — all implementation via `04-implementing` with
  `glm-5.3-flash:cloud`; no specialist model overrides.
- **Bypass legacy ceremony** — the plan-verification green-light cycle may be
  bypassed; the authoring instance self-checks with `slice-advancement` and
  returns.
- **Follow-ups go to the same idle agent** — within a phase, follow-up
  corrections route to the same `04-implementing` instance via `write_agent`.
- **No placeholder testing steps** — this is deterministic infrastructure work;
  red tests are written alongside implementation, not as separate ceremony
  steps.
- **Plan update at end of phase** — the orchestrator MUST update this tracker
  with status, evidence, and next boundary before advancing (simplified but not
  skipped).

## Acceptance criteria

- id: AC-001
  text: `validate-index.mjs` no longer references `DEFAULT_MAX_STALENESS_MS`,
  `over_age_paths`, or `OVER_AGE_FIX_HINT`; the `--max-age-ms` CLI flag is
  removed; no age-based staleness failure is emitted.
  validation: `npx jest --testPathPatterns=rag-index/__tests__/validate-index`
- id: AC-002
  text: `validate-index.mjs` emits a `family_fresh` map keyed by all 10
  `CORPUS_SOURCES` families, each with a `fresh` boolean and `stalePaths[]`;
  the top-level `pass` is the AND of all gated families
  (`completed-plan` excluded by default).
  validation: `npx jest --testPathPatterns=rag-index/__tests__/validate-index`
- id: AC-003
  text: `validate-index.mjs` deletion sweep compares indexed paths per family
  vs the glob of current files from `CORPUS_SOURCES` and deletes DB rows
  (documents + cascade chunks) for files no longer on disk. Comparison is
  performed on POSIX-normalized repo-relative paths (via `toRepoRelative` /
  `normalizeRepoPath`) on both the indexed `file_path` rows and the glob
  results, so no false deletions occur on Windows (where `path.sep='\'`).
  validation: `npx jest --testPathPatterns=rag-index/__tests__/validate-index`
  red_test: 'assert no false deletions when path.sep is mocked to `\` (or on
  Windows) — indexed rows with backslash separators must not be deleted'
- id: AC-003a
  text: `stalePaths[]` in each manifest family entry is sorted
  lexicographically (by family, then path) before the manifest is written,
  for replay-stable manifest output.
  validation: `npx jest --testPathPatterns=rag-index/__tests__/validate-index`
  red_test: 'assert manifest families[].stalePaths is lexicographically sorted
  across two runs with the same inputs'
- id: AC-004
  text: `rag-index/data/freshness-manifest.json` exists with one entry per
  family containing `fresh`, `stalePaths`, `lastReindex`, `maxSyncWaitMs`
  (plan: 60000), and `gated` (completed-plan: false, others: true);
  `artifacts/cortex-freshness-manifest.json` is removed. Manifest writes in
  `validate-index.mjs` use write-then-rename (atomic replace) so concurrent
  readers never observe a partial JSON file.
  validation: `node rag-index/validate-index.mjs --json` and inspect manifest
  red_test: 'assert concurrent write + read never throws JSON parse error
  (write-then-rename atomicity)'
- id: AC-004a
  text: `validate-index.mjs` retains a top-level `stale_paths` field (union
  of all families' stale paths) for backward compatibility, so
  `cortex-index.gate.runtime.mjs` `resolveFixHint` stale-plan branch
  continues to work after `over_age_paths` removal.
  validation: `npx jest --testPathPatterns=rag-index/__tests__/validate-index`
- id: AC-005
  text: `cortex-index.gate.mjs` exposes `evidence.family_fresh` as a map
  (family → boolean) and the gate `pass` is the AND of gated families only;
  `evidence.index_fresh` is retained as a backward-compatible alias equal to
  the AND of all `family_fresh` values. `completed-plan: gated=false` is
  hard-coded as the default so the gate does not depend on manifest presence.
  When `family_fresh` is undefined (manifest absent), the gate returns
  `pass: false` with fixHint "freshness manifest missing — run
  validate-index" (fails closed, NOT vacuously true).
  validation: `npx jest --testPathPatterns=cortex-index.gate`
  red_test: 'assert missing manifest → pass: false with missing-manifest
  fixHint (fails closed)'
- id: AC-006
  text: `auto-reindex.mjs` post-commit hook reindexes changed files matching
  `src/**/*.ts` (excl. test/spec/d.ts), `.github/skills/**/*.md`,
  `.github/agents/**/*.md`, `.github/copilot-instructions.md`, and
  `plans/*.plans.md`; non-matching files are ignored.
  validation: `npx jest --testPathPatterns=auto-reindex`
- id: AC-007
  text: `freshness-hooks.mjs` `DEFAULT_CHANGED_FILE_GLOBS` includes the extended
  glob set matching `auto-reindex.mjs`.
  validation: `npx jest --testPathPatterns=freshness-hooks`
- id: AC-008
  text: `reindex-plan-family.mjs` synchronously runs
  `build-index.mjs --files=<changed plan paths>` and returns
  `{ syncFresh: true, planFresh: true }` within 60s; dense embedding is queued
  in the background via `embed-index.mjs --files=…`.
  validation: `npx jest --testPathPatterns=reindex-plan-family`
- id: AC-009
  text: The pre-dispatch freshness hook waits up to `maxSyncWaitMs` (60s) for
  `families.plan.fresh === true`; on timeout it logs a warning and
  graceful-degrades (returns stale plan context with a `stale: true` flag)
  rather than blocking dispatch.
  validation: `npx jest --testPathPatterns=reindex-plan-family`
- id: AC-010
  text: `completed-plan` family is excluded from the default gate pass
  computation; its freshness is reported in `family_fresh` but does not fail
  the gate; manual `npm run index:archive` reindexes it.
  validation: `node scripts/agent-customization/gates/cortex-index.gate.mjs --json`
- id: AC-011
  text: `assemble-context.mjs` output chunks include an `indexed_at` field
  sourced from the `documents.indexed_at` column so cross-family queries can
  show staleness per result.
  validation: `npx jest --testPathPatterns=assemble-context`
- id: AC-012
  text: No backward-compatibility wrappers, dual-path code, or deferred cleanup
  remains — the old `artifacts/cortex-freshness-manifest.json`, the age gate,
  and the global `pass`-only emission are fully removed in the same step that
  introduces the new model. The `maxStalenessMs` field is removed from
  `SemanticIndexValidationInput` in `rag-index/validate-index.d.mts` and
  `family_fresh` is added to `SemanticIndexValidationResult`.
  validation: `grep -r "DEFAULT_MAX_STALENESS_MS\|over_age_paths\|cortex-freshness-manifest\|maxStalenessMs\|--max-age-ms\|OVER_AGE_FIX_HINT" rag-index/ scripts/ artifacts/ .github/` returns no hits
  sequencing_note: '`rag-index/README.md` (P0-S2-B) contains `--max-age-ms` references
  in its freshness docs; AC-012 grep covers `rag-index/` recursively and would flag
  README.md if run before P0-S2-B lands. AC-012 MUST be run after P0-S2-B completes
  (or scope the grep to exclude `rag-index/README.md` until P0-S2-B lands).'
- id: AC-013
  text: `rag-index/README.md` documents a reproducibility recipe: "To
  reproduce a known-good index state from scratch: `node
rag-index/build-index.mjs --force && node rag-index/embed-index.mjs
--force`."
  validation: `grep -r "build-index.mjs --force" rag-index/README.md` returns
  a hit
- id: AC-014
  text: `scripts/agent-customization/hooks/pre-dispatch-freshness-hook.mjs`
  is read-only — `updateManifest()` is removed; `manifestPath` points at
  `rag-index/data/freshness-manifest.json`; `readManifestTimestamp` reads
  `families.plan.lastReindex` (falling back to global `lastReindex`). The
  hook never writes the manifest (validator is the single writer).
  validation: `npx jest --testPathPatterns=pre-dispatch-freshness-hook`
  red_test: 'assert hook does not call writeFileSync on the manifest path;
  assert readManifestTimestamp returns null when manifest absent (not
  Infinity-stale via write path)'

## Non-goals

- No physical sharding of the Turso DB.
- No change to the `freshness.mjs` content-hash algorithm.
- No change to the `chunks_fts` schema or FTS5 trigger mechanics.
- No change to MCP tool signatures (`search_corpus`, `search_context`, etc.).
- No new DB schema columns (Phase 0–2); `indexed_at` already exists on
  `documents`.
- No CI pipeline changes beyond documenting the new manifest location.
- No change to `watch-plans.mjs` fs.watch mechanics (the new
  `reindex-plan-family.mjs` is the synchronous hook it invokes).

## Open assumptions

- **A1:** `session-start-index.mjs` touch pass becomes redundant once the age
  gate is removed (its purpose was to bump `indexed_at` to defeat the age gate).
  Resolution: simplify to incremental-build-only in Phase 0; remove the touch
  pass. If the touch pass has other consumers, keep it but document it as
  optional.
- **A2:** `cortex-index.gate.mjs` `evidence.index_fresh` is consumed by
  downstream gates/agents as a boolean. Resolution: retain it as a
  backward-compatible alias (AND of all gated `family_fresh`) to avoid
  breaking consumers.
- **A3:** The pre-dispatch freshness hook is a new integration point in the
  orchestrator dispatch flow. Resolution: Phase 2 wires it; the hook is
  optional and graceful-degrades, so no orchestrator change is required for
  the gate to function.

---

## Phase 0 — The fix (lowest risk) [DONE — compressed 2026-08-26]

**Objective:** Remove the age gate, add deletion sweep, per-family manifest,
per-family `fresh` emission, gate `family_fresh` map. No schema change, no
query-path change.

[DONE] Step P0-S1 — goal: remove age gate; add deletion sweep; per-family
manifest and fresh emission; gate `family_fresh` map. Slices P0-S1-A..E are
complete. Full step packet, per-slice records, and evidence are compressed
into `plans/RAG_Index_Freshness_Strategy.logs.md`
(§ "Phase 0 — Full detail (moved from plan tracker, 2026-08-26)").

## Phase 0 — Step 2: Session-start cleanup + README docs [DONE — compressed 2026-08-26]

[DONE] Step P0-S2 — goal: session-start touch pass simplification + README
docs for the new freshness model. Slices P0-S2-A and P0-S2-B are complete.
Full step packet and slice records are compressed into
`plans/RAG_Index_Freshness_Strategy.logs.md`
(§ "Phase 0 — Full detail (moved from plan tracker, 2026-08-26)").

---

## Phase 0 — Close-out [DONE — compressed 2026-08-26]

Phase 0 is complete and independently re-validated GREEN on 2026-08-26
(fresh session, real agents). The full AC audit table (AC-001–AC-014),
blockers/risks, and the close-out `PlanUpdate` are compressed into
`plans/RAG_Index_Freshness_Strategy.logs.md`
(§ "Phase 0 — Full detail (moved from plan tracker, 2026-08-26)").

Summary: all Phase 0 ACs PASS; AC-008/AC-009 (Phase 2) and AC-011 (Phase 3)
are pending out-of-scope work; the workflow-MCP binding was fixed and the
cortex-index gate passes end-to-end; artifacts were rebuilt (1880 docs,
24,935 chunks). Next boundary: Phase 1 re-validation.

---

<!-- fix-packet-P0-AC012-iteration-1 -->

## Fix packet: P0-AC012-iteration-1 [RESOLVED — compressed 2026-08-26]

Removed remaining literal references to the age gate and legacy manifest so
the AC-012 source-scope grep returns 0 hits. The full trigger, required
changes, validation commands, resolution evidence, and the artifact-
regeneration record are compressed into
`plans/RAG_Index_Freshness_Strategy.logs.md`
(§ "Fix packet P0-AC012-iteration-1 — full record").

---

## Phase 1 — Hook extension

**Objective:** Extend `auto-reindex.mjs` post-commit glob and
`freshness-hooks.mjs` watcher globs to cover the high-value corpus families, and
ensure the index builder, validator, entity graph, and hooks all recognize the
same families.

```yaml
step_packet:
step_id: 'P1-S1'
phase: 1
status: DONE
goal: 'Extend post-commit and watcher globs to src, skills, agents, copilot-instructions'
expansion: 'auto'
tdd_sequence:
  - red: 'rag-index/__tests__/auto-reindex.extended-glob.test.mjs — asserts non-plan changed files trigger reindex'
  - red: 'rag-index/__tests__/freshness-hooks.extended-glob.test.mjs — asserts extended globs match new families'
  - implement: '04-implementing'
  - green: '05-green-testing'
files_to_change:
  - 'rag-index/auto-reindex.mjs'
  - 'rag-index/freshness-hooks/freshness-hooks.mjs'
  - 'rag-index/build-index.mjs'
  - 'rag-index/validate-index.mjs'
  - 'rag-index/build-entity-graph.mjs'
  - 'rag-index/README.md'
  - 'rag-index/git-hooks/post-commit'
validation:
  - 'npx jest --testPathPatterns=auto-reindex'
  - 'npx jest --testPathPatterns=freshness-hooks'
  - 'node rag-index/validate-index.mjs --json'
research_artifact: plans/RAG_Index_Freshness_Strategy.research.md
evidence:
  gate_outputs:
    P1-S1-A:
      - 'shared-validation gate PASS (auto-reindex suites: 14/14 tests, build OK, lint OK)'
      - 'slice-advancement gate PASS (7/7 sub-gates)'
    P1-S1-B:
      - 'shared-validation gate PASS (freshness-hooks suites: 54/54 tests, build OK, lint OK)'
      - 'slice-advancement gate PASS'
      - 'specialist-review-severity: FULL -> review-coordinator -> api-contract-reviewer APPROVE'
  audit_findings:
    P1-S1-A:
      - status: PASS
        ac: AC-006
        reason: 'auto-reindex.mjs REINDEXABLE_FAMILIES matches plans/*.plans.md, src/**/*.ts (excl test/spec/d.ts), .github/skills/**/*.md, .github/agents/**/*.md, .github/copilot-instructions.md; resolveStalePlanFixHint remains plan-only; identifiers/log-path renamed; post-commit comment updated.'
        tests: 'rag-index/__tests__/auto-reindex.extended-glob.test.mjs — 3/3 passed under rag-index-mjs project'
    P1-S1-B:
      - status: PARTIAL
        ac: AC-007
        reason: 'freshness-hooks.mjs DEFAULT_CHANGED_FILE_GLOBS includes the new skill/agent/copilot-instruction globs and DEFAULT_IGNORE_GLOBS filters test/spec/d.ts; tests pass. However, the watcher globs are narrower than auto-reindex.mjs: .github/skills/**/SKILL.md does not cover references/assets .md files, and .github/agents/*.agent.md does not cover .github/agents/README.md. 24 corpus .md files are reindexed by post-commit but ignored by the post-write watcher hook.'
        tests: 'rag-index/__tests__/freshness-hooks.extended-glob.test.mjs — 2/2 passed under rag-index-mjs project'
        uncovered_files:
          count: 24
          examples:
            - '.github/skills/checkpointing-persistence/references/checkpoint-sources.md'
            - '.github/skills/educational-docs/assets/external-sources-and-media.md'
            - '.github/agents/README.md'
```

### Slice P1-S1-A — Extend post-commit hook glob [DONE — re-validated 2026-08-26]

**Files (≤3):** `rag-index/auto-reindex.mjs`,
`rag-index/__tests__/auto-reindex.extended-glob.test.mjs`

- Generalize `isPlanFile` → `isReindexableFile` **for the post-commit
  detection path only**: match `plans/*.plans.md`, `src/**/*.ts` (excl.
  `*.test.ts`, `*.spec.ts`, `*.d.ts`), `.github/skills/**/*.md`,
  `.github/agents/**/*.md`, `.github/copilot-instructions.md`.
- **`resolveStalePlanFixHint` in `auto-reindex.mjs` retains plan-only
  filtering** — its name and contract are plan-specific. The
  `isPlanFile` → `isReindexableFile` generalization applies only to the
  post-commit detection path, not to `resolveStalePlanFixHint`, which
  continues to filter via `isPlanFile`.
- `buildReindexCommands` produces `--files=` args for all matched paths (not
  just `.plans.md`).
- Rename internal `reindexChangedPlans`/`detectChangedPlanFiles` identifiers
  to `reindexChangedFiles`/`detectChangedFiles` and update log path to
  `auto-reindex.log` to reflect the generalized corpus behavior. Keep
  exported `reindexChangedFiles` for callers.
- Update `git-hooks/post-commit` comment to describe general corpus reindex.
- Red tests assert: a changed `src/foo.ts` triggers reindex; a changed
  `src/foo.test.ts` does not; a changed `.github/skills/foo/SKILL.md`
  triggers reindex; `resolveStalePlanFixHint` still filters plan-only.

### Slice P1-S1-B — Extend watcher globs [DONE — re-validated 2026-08-26; fix packet P1-S1-B-iteration-1 RESOLVED]

**Files (≤3):** `rag-index/freshness-hooks/freshness-hooks.mjs`,
`rag-index/__tests__/freshness-hooks.extended-glob.test.mjs`

- Extend `DEFAULT_CHANGED_FILE_GLOBS` to include:
  `.github/skills/**/SKILL.md`, `.github/agents/*.agent.md`,
  `.github/copilot-instructions.md`.
- `src/**/*.ts` is already present; add explicit `.test.ts`, `.spec.ts`,
  `.d.ts` exclusions in the flush filter so they never reach the incremental
  builder as family `unknown`.
- Add the new `skill`, `agent`, and `copilot-instructions` families to
  `FAMILY_RULES` so family inference works for these paths.
- Red tests assert: the new globs match the extended family paths;
  test/spec/d.ts files are filtered out before the incremental build.

---

<!-- fix-packet-P1-S1-B-iteration-1 -->

## Fix packet: P1-S1-B-iteration-1 [RESOLVED — 2026-08-26]

```yaml
fix_packet:
  fix_packet_id: fix-packet-P1-S1-B-iteration-1
  slice_id: P1-S1-B
  status: RESOLVED
  source: '02-researching P1-S1 audit (2026-08-26): AC-006 PASS, AC-007 PARTIAL'
  observations:
    - 'freshness-hooks.mjs DEFAULT_CHANGED_FILE_GLOBS uses .github/skills/**/SKILL.md and .github/agents/*.agent.md; auto-reindex.mjs post-commit set uses .github/skills/**/*.md and .github/agents/**/*.md — 23 skill .md + 1 agent .md files are reindexed by post-commit but ignored by the post-write watcher (silent index-drift risk)'
    - 'AC-007 requires DEFAULT_CHANGED_FILE_GLOBS to include the extended glob set matching auto-reindex.mjs; current watcher set does not match'
  required_changes:
    - 'Widen freshness-hooks.mjs DEFAULT_CHANGED_FILE_GLOBS to .github/skills/**/*.md and .github/agents/**/*.md (matching auto-reindex REINDEXABLE_FAMILIES); keep test/spec/d.ts exclusions intact'
    - 'Update rag-index/__tests__/freshness-hooks.extended-glob.test.mjs to assert the widened globs (e.g., .github/skills/foo/README.md and .github/agents/README.md now match and reach the incremental builder)'
  validation:
    - 'npm run jest:mjs -- --testPathPatterns=freshness-hooks --no-coverage'
    - 'npm run jest:mjs -- --testPathPatterns=auto-reindex --no-coverage'
  followup_note: none
  resolution:
    date: '2026-08-26'
    changed_files:
      - 'rag-index/freshness-hooks/freshness-hooks.mjs'
      - 'rag-index/__tests__/freshness-hooks.extended-glob.test.mjs'
      - 'rag-index/freshness-hooks/freshness-hooks.test.mjs'
    changes_applied:
      - 'DEFAULT_CHANGED_FILE_GLOBS widened to .github/skills/**/*.md and .github/agents/**/*.md; DEFAULT_IGNORE_GLOBS test/spec/d.ts exclusions untouched'
      - 'extended-glob test asserts widened globs and verifies .github/skills/foo/README.md + .github/agents/README.md now reach the incremental builder'
      - 'owner-local freshness-hooks.test.mjs glob assertions updated to the widened set'
    evidence:
      - 'PASS: freshness-hooks 55/55 tests (2 suites)'
      - 'PASS: auto-reindex 15/15 tests (2 suites; console.error output is an expected mocked-failure assertion)'
```

---

## Phase 1 — Close-out

**Status:** [DONE — independently re-validated GREEN 2026-08-26]

Phase 1 re-validation completed on a fresh session with real agents, per the
restart protocol: `02-researching` audited P1-S1 (AC-006 PASS; AC-007 PARTIAL
— watcher globs narrower than post-commit globs), fix packet
`P1-S1-B-iteration-1` was resolved by `04-implementing` (widened
`freshness-hooks.mjs` `DEFAULT_CHANGED_FILE_GLOBS` to
`.github/skills/**/*.md` + `.github/agents/**/*.md`, matching
`auto-reindex.mjs` `REINDEXABLE_FAMILIES`), and `05-green-testing`
green-validated P1-S1 (auto-reindex 15/15, freshness-hooks 55/55,
`validate-index.mjs --json` ok:true pass:true, shared-validation gate PASS
132/132 + build + lint, slice-advancement 7/7). Prior shell-level evidence is
superseded by this independent validation.

```yaml
PlanUpdate:
  phase: 1
  status: DONE
  note: 'Phase 1 re-validated GREEN 2026-08-26 on a fresh session. AC-006 PASS; AC-007 PASS after fix-packet-P1-S1-B-iteration-1 (watcher globs aligned with post-commit globs). Known environment risk: workflow MCP server not bound to the active plan path (workflow_mcp_alive:false); direct plan-file fallback works. Phase 2 is unblocked.'
  next_boundary:
    phase: 2
    step: P2-S1
    resume: 'Phase 2 — Sync plan hook: dispatch 03/04 for P2-S1-A (reindex-plan-family.mjs synchronous hook) then P2-S1-B (waitForPlanFresh pre-dispatch integration), then 05-green-testing. Phase 1 gate is cleared (GREEN 2026-08-26).'
```

---

## Phase 2 — Sync plan hook [DONE — independently GREEN 2026-08-27]

**Objective:** New `reindex-plan-family.mjs` synchronous post-save plan hook
with 60s budget + graceful degrade; exclude `completed-plan` from default gate.

**Phase 1 was re-validated by the real `02-researching` and `05-green-testing`
agents on 2026-08-26 (fresh session) — this phase is unblocked. Use
`npm run jest:mjs -- --testPathPatterns=<pattern>` for validation commands;
the plan packet's bare `npx jest` commands require the ESM flags for `.mjs`
tests.**

```yaml
step_packet:
  step_id: 'P2-S1'
  phase: 2
  goal: 'Synchronous post-save plan hook with 60s budget, graceful degrade, completed-plan gate exclusion'
  expansion: 'auto'
  tdd_sequence:
    - red: 'rag-index/__tests__/reindex-plan-family.test.mjs — asserts sync BM25 within 60s, background dense queued, graceful degrade on timeout'
    - implement: '04-implementing'
    - green: '05-green-testing'
  files_to_change:
    - 'rag-index/reindex-plan-family.mjs (new)'
    - 'rag-index/watch-plans.mjs (wire onChange to reindex-plan-family)'
    - 'scripts/agent-customization/gates/cortex-index.gate.mjs (completed-plan exclusion already in P0; verify)'
  validation:
    - 'npx jest --testPathPatterns=reindex-plan-family'
    - 'npx jest --testPathPatterns=watch-plans'
    - 'node scripts/agent-customization/gates/cortex-index.gate.mjs --json'
  evidence:
    gate_outputs: {}
```

### Slice P2-S1-A — `reindex-plan-family.mjs` synchronous hook [DONE — GREEN 2026-08-27]

**Files (≤3):** `rag-index/reindex-plan-family.mjs`,
`rag-index/__tests__/reindex-plan-family.test.mjs`,
`rag-index/watch-plans.mjs`

- New `reindex-plan-family.mjs` exports `reindexPlanFamily(changedPaths,
{ maxSyncWaitMs })`:
  - Synchronously runs `build-index.mjs --files=<changed plan paths>` (BM25).
  - Returns `{ syncFresh: true, planFresh: true }` on success.
  - Queues `embed-index.mjs --files=<changed plan paths>` in the background
    (non-blocking).
  - On timeout (> `maxSyncWaitMs`, default 60s): logs warning, returns
    `{ syncFresh: false, planFresh: false, stale: true, reason: 'timeout' }`.
- `watch-plans.mjs` `onChange` callback invokes `reindexPlanFamily` instead of
  the current direct build call (if any).
- Red tests assert: sync BM25 completes within 60s; dense is queued
  (mocked); timeout returns `stale: true` without throwing.

### Slice P2-S1-B — Pre-dispatch freshness hook integration [DONE — GREEN 2026-08-27]

**Files (≤3):** `rag-index/reindex-plan-family.mjs`,
`rag-index/__tests__/reindex-plan-family.test.mjs`,
`rag-index/data/freshness-manifest.json`

- Add `waitForPlanFresh({ maxSyncWaitMs })` to `reindex-plan-family.mjs`:
  polls `rag-index/data/freshness-manifest.json`
  `families.plan.fresh` up to `maxSyncWaitMs`; on timeout graceful-degrades
  with a warning and returns `{ planFresh: false, stale: true }`.
- **`waitForPlanFresh` must catch JSON parse errors** (from a concurrent
  atomic write-then-rename in flight) and treat them as "not yet fresh"
  rather than throwing. This complements the atomic-replace mandate in
  P0-S1-B.
- This is the pre-dispatch integration point: orchestrators call
  `waitForPlanFresh` before dispatching plan-dependent work; on stale they
  proceed with a `stale: true` warning rather than blocking.
- Red tests assert: `waitForPlanFresh` returns `planFresh: true` when
  manifest says fresh; returns `stale: true` on timeout; **does not throw
  when manifest is mid-write (concurrent write + poll red test)**.

## Phase 2 — Close-out [DONE — independently GREEN 2026-08-27]

Phase 2 executed under the plan's pragmatic mandates (broad slice: one
`04-implementing` dispatch covering P2-S1-A + P2-S1-B with red tests written
alongside implementation; specialist-review ceremony bypassed; shared-validation
and green-testing NOT bypassed and both PASS).

```yaml
PlanUpdate:
  phase: 2
  status: DONE
  changes: 'New rag-index/reindex-plan-family.mjs (reindexPlanFamily: sync BM25 within 60s budget, background dense queue, graceful stale degrade; waitForPlanFresh: parse-tolerant manifest polling with timeout degrade; CLI entry). watch-plans.mjs onChange now defaults to the hook via createReindexPlanOnChange factory. 21 new tests + 3 watcher tests.'
  evidence:
    - 'shared-validation.gate.mjs PASS (39/39 tests, build OK, lint OK; watch-plans.test.ts 1/1)'
    - '05-green-testing GREEN: reindex-plan-family 21/21, watch-plans 18/18, tsc --noEmit clean, eslint clean, code-coverage gate PASS, slice-advancement 7/7 severity FULL'
    - 'End-to-end dogfood: node rag-index/reindex-plan-family.mjs --files=plans/RAG_Index_Freshness_Strategy.plans.md -> {syncFresh:true, planFresh:true}; family_fresh.plan.fresh:true after reindex'
    - 'cortex-index.gate.mjs workflow_mcp_alive:false — documented environment limitation, not a code defect'
  removals: []
  next_boundary:
    phase: 3
    step: P3-S1
    resume: 'Phase 3 — Optional hardening: slice P3-S1-A (assemble-context.mjs indexed_at tagging per AC-011) then P3-S1-B (per-family max_age_ms sanity signal for missing files only). Phase 2 gate is cleared (GREEN 2026-08-27).'
```

---

## Phase 3 — Optional hardening [DONE — compressed 2026-08-27]

**Objective:** Per-family `max_age_ms` as secondary sanity signal for
deleted/missing files only; `indexed_at` tagging in `assemble-context.mjs`.

**Outcome:** Slices P3-S1-A (`indexed_at` tagging in `assemble-context.mjs`,
AC-011) and P3-S1-B (optional per-family `max_age_ms` deletion-sweep sanity
signal in `validate-index.mjs`, warning-only) were implemented RED-first and
independently GREEN-validated 2026-08-27. Step packet P3-S1, slice records,
and per-slice implementation evidence are compressed into
`plans/RAG_Index_Freshness_Strategy.logs.md`
(§ "Phase 3 — Optional hardening (moved from plan tracker, 2026-08-27)").

---

## Traceability

| AC      | Phase | Slice                     | Files                                                                                                                   |
| ------- | ----- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| AC-001  | 0     | P0-S1-A/E                 | `validate-index.mjs`, `validate-index.d.mts`, `validate-index.test.mjs`, `embed-index.test.ts`                          |
| AC-002  | 0     | P0-S1-B                   | `validate-index.mjs`, `freshness-manifest.json`                                                                         |
| AC-003  | 0     | P0-S1-A                   | `validate-index.mjs`                                                                                                    |
| AC-003a | 0     | P0-S1-B                   | `validate-index.mjs`, `freshness-manifest.json`                                                                         |
| AC-004  | 0     | P0-S1-B                   | `freshness-manifest.json`, `artifacts/…` (deleted)                                                                      |
| AC-004a | 0     | P0-S1-B                   | `validate-index.mjs`                                                                                                    |
| AC-005  | 0     | P0-S1-C/E                 | `cortex-index.gate.mjs`, `cortex-index.gate.runtime.mjs`, `cortex-index.gate.runtime.direct.test.mjs`                   |
| AC-006  | 1     | P1-S1-A                   | `auto-reindex.mjs`                                                                                                      |
| AC-007  | 1     | P1-S1-B                   | `freshness-hooks.mjs`                                                                                                   |
| AC-008  | 2     | P2-S1-A                   | `reindex-plan-family.mjs`, `watch-plans.mjs`                                                                            |
| AC-009  | 2     | P2-S1-B                   | `reindex-plan-family.mjs`, `freshness-manifest.json`                                                                    |
| AC-010  | 0     | P0-S1-C                   | `cortex-index.gate.mjs`                                                                                                 |
| AC-011  | 3     | P3-S1-A                   | `assemble-context.mjs`                                                                                                  |
| AC-012  | 0     | P0-S1-A/B/C/D/E + P0-S2-B | all Phase 0 core files + `repo-cortex-workflow/SKILL.md` + `rag-index/README.md` (P0-S2-B must land before AC-012 grep) |
| AC-013  | 0     | P0-S2-B                   | `rag-index/README.md`                                                                                                   |
| AC-014  | 0     | P0-S1-D                   | `pre-dispatch-freshness-hook.mjs`                                                                                       |

## Risks

- **R1:** Downstream gates/agents consume `evidence.index_fresh` as a boolean.
  Mitigation: retain it as a backward-compatible alias (A2).
- **R2:** `session-start-index.mjs` touch pass may have consumers beyond the
  age gate. Mitigation: A1 — verify before removing; keep as optional if
  needed.
- **R3:** Deletion sweep adds DB writes to the validate path. Mitigation:
  sweep runs only when missing paths are detected; no-op when all files
  present.
- **R4:** Synchronous 60s plan hook may exceed budget on large plan sets.
  Mitigation: graceful degrade returns `stale: true` rather than blocking;
  orchestrator proceeds with warning.
- **R5:** Extending post-commit hook to `src/**/*.ts` may slow commits on
  large changesets. Mitigation: `--files=` targets only changed paths;
  build-index skips unchanged files via freshness proof.
- **R6 (determinism rung — Level 2, content-deterministic):** The manifest
  records `indexed_at = Date.now()` as intentionally non-deterministic
  metadata. Freshness equality is content-hash-only (`sha256` + mtime +
  size); `indexed_at` is excluded from freshness equality. AC-011 tests
  assert `indexed_at` presence but not a specific value. Manifest
  `stalePaths[]` sorting (AC-003a) and atomic write-then-rename (AC-004)
  ensure replay-stable manifest output apart from the `indexed_at` field.
- **R7 (manifest write concurrency):** The pre-dispatch freshness hook must
  not write the manifest (determinism reviewer). The validator is the single
  writer; the hook is read-only (AC-014). `waitForPlanFresh` catches JSON
  parse errors from in-flight atomic writes (P2-S1-B).

## Implementation phases

### Phase 0 — Re-validation frontier [DONE — compressed 2026-08-26]

The re-validation step packet and outcome are compressed into
`plans/RAG_Index_Freshness_Strategy.logs.md`
(§ "Phase 0 — Independent re-validation frontier").

**Outcome (2026-08-26, fresh session):** audit → remediation → green loop
complete. `02-researching` PASS on all Phase 0 ACs; `00-helping` resolved the
workflow-MCP binding; `04-implementing` rebuilt artifacts; `05-green-testing`
GREEN. Acceptance criterion met.

## Validation gates

## Latest validation evidence

All Phase 0 and Phase 1 implementation slices are recorded as implementation-complete, but the prior validation was produced via shell-level fallback after the CLI task-tool registry became corrupted. The detailed per-slice evidence has been compressed into `plans/RAG_Index_Freshness_Strategy.logs.md`. All remaining Phase 0 detail (step packets P0-S1/P0-S2, slice records, close-out AC audit, fix packet P0-AC012-iteration-1, and the Phase 0 re-validation step packet) was moved to that log on 2026-08-26; this plan retains compact [DONE] markers only.

- **Phase 0 — The fix:** **[DONE — independently GREEN 2026-08-26]** (`02-researching` audit: 0 implementation defects, all Phase 0 ACs PASS on real code; `05-green-testing`: jest 9/9 validate-index, 55/55 cortex-index.gate, 34/34 pre-dispatch-freshness-hook, 10/10 session-start-index; `validate-index.mjs --json` ok:true pass:true; `cortex-index.gate.mjs --json` pass:true incl. `workflow_mcp_alive: true`; AC-012 source-scope grep clean; AC-013 recipe present; code-coverage gate pass:true; slice-advancement 7/7 on P0-S1 and P0-S2)
- **Phase 1 — Hook extension:** **[DONE — independently GREEN 2026-08-26]** (`02-researching` audit: AC-006 PASS, AC-007 PARTIAL → fix packet `P1-S1-B-iteration-1` resolved by `04-implementing` (watcher globs widened to match post-commit set); `05-green-testing` GREEN: auto-reindex 15/15, freshness-hooks 55/55, `validate-index.mjs --json` ok:true pass:true (1880 docs, 24,951 chunks), shared-validation PASS 132/132 + build + lint, slice-advancement 7/7 severity FULL; only `cortex-index.gate.mjs` `workflow_mcp_alive:false` fails — environment (workflow MCP not bound to active plan path), not a code defect)
- fix-loop: P1-S1-B iteration 1 status=passed (fix packet P1-S1-B-iteration-1 resolved 2026-08-26: watcher globs widened to `.github/skills/**/*.md` + `.github/agents/**/*.md` matching auto-reindex; freshness-hooks 55/55 PASS, auto-reindex 15/15 PASS)
- **Phase 1 — P1-S1 green validation (2026-08-26):** `npm run jest:mjs -- --testPathPatterns=auto-reindex` 15/15 PASS; `npm run jest:mjs -- --testPathPatterns=freshness-hooks` 55/55 PASS; `node rag-index/validate-index.mjs --json` ok:true pass:true after targeted reindex of stale plan/research files; `shared-validation.gate.mjs` PASS (132/132 tests, build OK, lint OK); `slice-advancement.gate.mjs` PASS (7/7 sub-gates, severity FULL, specialist review confirmed). `cortex-index.gate.mjs` reports `workflow_mcp_alive:false` because the workflow MCP server is not bound to the active plan path (environment/tooling, not a code defect); corpus index is fresh and all families fresh.
- **Phase 2 — Sync plan hook:** **[DONE — independently GREEN 2026-08-27]** — `05-green-testing` green-validated P2-S1 (slices P2-S1-A and P2-S1-B) against AC-008/AC-009: `npm run jest:mjs -- --testPathPatterns=reindex-plan-family` 21/21 PASS; `--testPathPatterns=watch-plans` 18/18 PASS (39/39 total); `npx tsc --noEmit` clean; eslint clean on all changed files; `shared-validation.gate.mjs` PASS (39/39 tests, build OK, lint OK); `code-coverage.gate.mjs` PASS (100% on touched `scripts/agent-customization/gates/cortex-index.gate.mjs`); `slice-advancement.gate.mjs` PASS (7/7 sub-gates, severity FULL). End-to-end dogfood (`node rag-index/reindex-plan-family.mjs --files=plans/RAG_Index_Freshness_Strategy.plans.md`) returned `{syncFresh:true, planFresh:true}` and restored `family_fresh.plan.fresh:true` after reindexing. `cortex-index.gate.mjs` still reports `workflow_mcp_alive:false` because the workflow MCP server is not bound to the active plan path (environment/tooling, not a code defect); all corpus families are fresh.
- **Phase 3 — P3-S1-A indexed_at tagging (2026-08-27):** [DONE — GREEN] 04-implementing single-slice dispatch; `assemble-context.mjs` enrichment JOIN maps `documents.indexed_at` onto every chunk (AC-011); red-first (4 failed pre-implementation, 5/5 pass post). Full per-slice evidence compressed into `plans/RAG_Index_Freshness_Strategy.logs.md` (§ "Phase 3 — Optional hardening").
- **Phase 3 — P3-S1-B per-family `max_age_ms` sanity signal (2026-08-27):** [DONE — GREEN] 04-implementing single-slice dispatch; optional per-family `max_age_ms` manifest field (default null), warning-only deletion-sweep signal, `warnings` result field, `validate-index.d.mts` + manifest schema evolution; red-first (18/18 failed pre-implementation, 18/18 pass post). Full per-slice evidence compressed into `plans/RAG_Index_Freshness_Strategy.logs.md` (§ "Phase 3 — Optional hardening").
- **Phase 3 — Optional hardening:** [DONE — independently GREEN 2026-08-27] P3-S1-A `assemble-context` 116/117 PASS (1 pre-existing skip); P3-S1-B `validate-index` 45/45 PASS; `validate-index.mjs --json` ok:true pass:true after plan-family reindex; `slice-advancement.gate.mjs` 7/7 PASS (severity FULL, specialist review confirmed).
- **Phase 3 — 05-green-testing independent green (2026-08-27):** `assemble-context` 116/117 PASS (1 pre-existing skip; indexed-at 5/5); `validate-index` 45/45 PASS (max-age-sanity 18/18); `validate-index.mjs --json` ok:true pass:true (all 10 families fresh, no warnings after plan-family reindex); `slice-advancement.gate.mjs` direct run PASS 7/7 sub-gates, severity FULL (code-coverage N/A — no `src/` or `scripts/agent-customization/` files touched). MCP tooling errors (`get_active_validation_allowlist` plan-status, `slice-advancement` timeout) were worked around via direct shell/gate-script runs. Full evidence in the log (§ "Phase 3 — Optional hardening").

**Pragmatic bypass note:** The plan-verification green-light cycle is bypassed for this plan per the `## Mandates` section (pragmatic mode authorized: "Bypass legacy ceremony — the plan-verification green-light cycle may be bypassed; the authoring instance self-checks with `slice-advancement` and returns").

**Gate baseline (2026-08-26 re-validation pass):**

- `plan-sync`: PASS · `slice-advancement`: PASS (7/7 on P0-S1 and P0-S2) · `plan-readiness`: PASS · `plan-command-lint`: PASS
- `cortex-index` gate: **end-to-end PASS** (`pass:true`, `workflow_mcp_alive:true`, all 10 families fresh, `fixHint:null`)
- `code-coverage` gate: PASS (100% on touched `scripts/agent-customization/` files)
- Semantic snapshot: **rebuilt 2026-08-26** (1880 docs, 24,935 chunks) — the earlier ~6.5-day-staleness note is obsolete; the pre-dispatch freshness hook runs clean.

Restart protocol before resuming:

1. ✅ Restart CLI / VS Code — done (fresh session; task-tool registry healthy: 00/02/04/05 agents all dispatched successfully).
2. ✅ Refresh semantic snapshot — done (forced full reindex + snapshot rebuild during Phase 0 close-out).
3. ✅ Agent reachability — verified by live dispatches this session.
4. ✅ Repo-side gate baseline — green (see above).
5. ✅ Re-validate **Phase 0** — audit + green both PASS 2026-08-26.
6. ✅ Re-validate **Phase 1** from the top — done 2026-08-26 (02-researching audit → fix-packet-P1-S1-B-iteration-1 resolved → 05-green-testing GREEN).
7. ✅ Phase 1 gate cleared — done 2026-08-26.
8. ✅ Phase 2 executed and independently GREEN 2026-08-27 — 04-implementing (P2-S1-A + P2-S1-B, broad slice) → shared-validation PASS → 05-green-testing GREEN. **Next boundary: Phase 3, step P3-S1 (slices P3-S1-A then P3-S1-B).**

**Plan-readiness marker:** `green-light: true` — recorded per pragmatic-bypass authorization in `## Mandates`.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history. Load context via Cortex MCP and any declared pre_execute_hook/get_slice_context.

The RAG Index Freshness Strategy plan: **ALL PHASES [DONE]** — independently re-validated GREEN. Phase 0/1 on 2026-08-26; Phase 2 on 2026-08-27; Phase 3 on 2026-08-27. Phase 3 green validation (05-green-testing): `assemble-context` 116/117 PASS, `validate-index` 45/45 PASS, `validate-index.mjs --json` ok:true pass:true, `slice-advancement` 7/7 PASS. No remaining work; plan is complete.

Immediate next steps: none — archive this plan per tracker-handoff conventions if desired.

Known environment risk: the workflow MCP server may be unbound from the active plan path (`workflow_mcp_alive:false` in the cortex-index gate); direct plan-file reads are the working fallback, and reindexing the plan file after edits restores index freshness.

Prior per-slice evidence and Phase 0/3 detail (close-out AC audit, fix packet P0-AC012-iteration-1, Phase 0 re-validation record, Phase 3 step packet + slice records) live in plans/RAG_Index_Freshness_Strategy.logs.md; the plan file carries compact [DONE] markers plus '## Latest validation evidence' above. The plan is complete — no further work remains; the only optional follow-up is archiving the plan/log pair to plans/completed/ per tracker-handoff conventions.
```
