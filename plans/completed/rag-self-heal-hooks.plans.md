# RAG / Cortex Self-Healing Hooks

**Status:** [DONE]

Automatic detection-and-repair safety net for Cortex/RAG degradation, staleness, and unavailability, with model-facing guidance and zero-user-intervention repair using the existing index automation.

<details>
<summary>Authoring note (workflow contract) — expand for how this plan is structured</summary>

- Phase → step → slice packets follow the **new step-packet format** validated by `scripts/agent-customization/gates/step-packet.gate.mjs` (goal/expansion/tdd_sequence enums, slice goal ordering).
- Only the current `[WIP]` phase/step packets are gate-validated; `[PLANNED]` packets are authored in the same format so future phases activate without rewrites.
- Step packets reference the consolidated Tier-1 gate `slice-advancement` (plan-sync + step-packet + plan-slice-quality + plan-command-lint). Sub-gates must never be run individually as "validation"; use `neataptic-gate-mcp:run_gate_check` with `gate: 'slice-advancement'` and `args: { slice-id, changed-files }`.
- **`.mjs` test convention:** hook/script/server tests are colocated `*.test.mjs` files running under the `rag-index-mjs` / `mcp-semantic-mjs` / `agent-customization-mjs` jest projects. Validation commands use `npm run jest:mjs -- --testPathPatterns=<regex> --runInBand --no-coverage` (plural `--testPathPatterns`; bare `npx jest` without the ESM flags fails for `.mjs` suites).
- This is the planning-author pass. Green light is recorded only by the independent verification pass (Phase 1, Step 03).

</details>

## Purpose

Today Cortex hybrid search silently degraded: `search_corpus` returned `dense_degraded: true`, `dense_reason: "Embeddings are incomplete: expected 24957, found 0"`, `dense_state: "model-only"`. The BM25 corpus index was fresh (24,957 chunks, 10 families) but the dense embeddings store was empty, so every dense/hybrid query fell back to BM25-only with degraded recall. Detection and remediation were manual (`node rag-index/validate-index.mjs --json`, then `npm run index:prewarm`). This plan makes that entire failure class **self-heal automatically at call time**, with the model clearly informed about what is happening, how long repair takes, and when to pause and ask the user.

## Scope

**In scope**

- Cheap call-time detection of degraded/stale/unusable Cortex state on search surfaces (`search_corpus`, `search_context`, `index_stats`).
- A health-state file + repair lock under `rag-index/data/` with cooldown, exponential backoff, and a max-attempts ceiling.
- Async, detached background repair that reuses existing scripts (never duplicates their logic).
- Model-facing guidance blocks appended to search responses and to the facade's server-unavailable path.
- Empirical confirmation of the open mechanism questions before implementation begins.

**Explicitly out of scope (non-goals)**

- No new periodic updater / scheduler. The mechanism is an event-driven safety net triggered by actual call-time signals only.
- No re-registration of the heavy manual-only hooks (`session-start-cortex-mcp-preflight.mjs`, `refresh-cortex-after-write.mjs`); existing SessionStart/PostToolUse hook wiring stays unchanged.
- No changes to BM25 ranking, embedding model selection, Turso sync wiring, or corpus families.
- No new hook registrations. Option A (environment hooks matching MCP tool names) is demoted pending harness-support verification (see Research findings to confirm, Step 02).

## Architecture decisions

### Mechanism choice (A / B / C evaluation)

| Option                                                                   | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Verdict                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A — Environment hooks (PreToolUse/PostToolUse matching MCP tool names)   | Hooks registered in `.github/hooks/cortex-refresh.json` are `{command}`-only entries; per-tool filtering happens inside the script (see `post-write-reindex-hook.mjs`). Whether this harness fires hooks for MCP-routed tool calls (`cortex-cortex`) is **unverified**; if it does not fire, the safety net silently never triggers — the exact failure class this plan exists to kill. Hooks also run on every matching call (hot-path overhead) and PostToolUse result inspection support is unconfirmed.                                                 | **Demoted** — optional additive layer only after Step 02 empirically proves MCP-tool matching + context injection/block-reason support.                                                                                                                                                 |
| B — Server-internal self-heal                                            | Detection lives exactly where the `dense_degraded`/`dense_reason`/`dense_state` fields are already produced (`scripts/mcp-semantic/tools/search-corpus.mjs`, `search-context.mjs`; `index_stats` in `scripts/mcp-semantic/repo-cortex-mcp.mjs`). One shared guard module covers every consumer (facade tool, gates, future clients) with **zero registration**. The model-facing instruction block rides inside the structured response the model already parses. Async detached repair spawn follows the proven `pre-dispatch-freshness-hook.mjs` pattern. | **Recommended.** Requires care: fire-and-forget spawn gated by cooldown/lock so a burst of searches during repair never spawns N repairs; and the process-lifetime `cachedDenseReadiness` memo must be invalidated after repair or a running server keeps answering "degraded" forever. |
| C — Hybrid (server-internal + shared state + hooks for non-MCP surfaces) | The shared health-state file is genuinely valuable for cross-surface coordination (session-start hook and gates can consult it later, read-only). But hooks for MCP-call detection add nothing over B and reintroduce A's risks.                                                                                                                                                                                                                                                                                                                            | **Partially adopted** — adopt C's shared state file + consult-only readers; adopt **no** new hook registrations.                                                                                                                                                                        |

**Recommendation: Option B with Option C's shared state file ("server-internal core, shared-state coordination").** Search-time degraded handling, cooldown/backoff bookkeeping, repair triggering, and model guidance all live in the MCP server process where the signal originates and where the response is assembled; the state/lock files under `rag-index/data/` make the behavior observable and let non-MCP surfaces (session-start hook, gates, humans) consult repair status without duplicating probes. This is the most reliable zero-intervention design: no harness-configuration dependency can silently void the safety net, guidance is guaranteed to reach the model in the tool response it is already reading, and reuse of existing repair scripts stays compile-time visible in one orchestrator module.

> User: Approved recommendation

### Detection signal inventory (codified by the guard module)

| Source                                                                             | Signal                                                             | Meaning                                                               | Probe cost                                                                                      |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `search_corpus` / `search_context` responses                                       | `dense_degraded: true`, `dense_reason`, `dense_state: cold         | model-only`                                                           | Dense path unusable; BM25-only fallback active                                                  | Free — already computed per call via `checkDenseReadiness()`                                                                           |
| `rag-index/dense-readiness.mjs` `checkDenseReadiness()`                            | `state: cold                                                       | model-only                                                            | warm`, `chunk_count`, `embedding_count`                                                         | Cold = ONNX model cache absent; model-only = model present, embeddings missing/incomplete (today's incident: expected 24,957, found 0) | Cheap — file-stat + sqlite row count; memoized per process (must invalidate after repair) |
| `index_stats`                                                                      | `total_chunks` vs embeddings row-count mismatch                    | Independent confirmation of embedding-store emptiness                 | Cheap — same sqlite reads                                                                       |
| `node rag-index/validate-index.mjs --json`                                         | `stale_paths`, `missing_paths`, per-family `family_fresh` booleans | BM25 corpus staleness (stale-result symptom: hits from deleted files) | Heavy — **never** run synchronously on the search path; run inside the repair orchestrator only |
| `scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json`                    | MCP liveness                                                       | Real server process spawn/liveness probe                              | Runs in Phase-1 hook and gates; consulted read-only by the orchestrator after repair            |
| Facade lazy-spawn failure (`scripts/agent-customization/mcp/lazy-facade-core.mjs`) | Spawn error / `tools/call` transport failure                       | MCP "dead / no results at all" — the only hard-block case             | Free — existing error path, currently a one-line `fallbackHint`                                 |

### State and lock files (new artifacts, following existing conventions)

Freshness state today lives in `rag-index/data/freshness-manifest.json` (single writer: `validate-index.mjs`; atomic persistence conventions). The new artifacts follow the same shape:

- `rag-index/data/cortex-self-heal-state.json` — **single writer: `cortex-self-heal.mjs` (the repair orchestrator); all other surfaces are read-only.** Written atomically (write-then-rename). Contents: `version`, `last_probe {state, dense_reason, chunk_count, embedding_count, probed_at}`, `cooldown {cooldown_s, last_trigger_at}`, `backoff {attempts_in_window, window_started_at, backoff_factor, next_allowed_at}`, `max_attempts`, `in_flight {pid, started_at, attempt, reason, est_duration_min} | null`, `history` (last 5 repair runs with duration and result, used to estimate future durations).
- `rag-index/data/cortex-self-heal.repair.lock` — mutual exclusion across every repair-capable surface. Created atomically (`fs.open` with `'wx'`) by the guard at the moment it decides to trigger repair, then owned by the detached orchestrator process. Lock record: `{pid, started_at, heartbeat_at, attempt, reason}`. Stale when `heartbeat_at` is older than `CORTEX_SELFHEAL_LOCK_STALE_S` **or** the owning pid is no longer alive; a stale lock is reclaimed by the next caller after a single forced `unlink` retry.
- `.gitignore` must be extended so both files (and `rag-index/data/*.lock`) are never committed — current ignore patterns cover `*.sqlite`, `hook-context/*.json`, and `mcp-session-override.json` but not new JSON state files.

Default knobs (env-overridable, same convention as `CORTEX_GRACE_WINDOW_S` / `CORTEX_STALENESS_THRESHOLD_S` in `pre-dispatch-freshness-hook.mjs`):

| Env var                                   | Default | Meaning                                                                               |
| ----------------------------------------- | ------- | ------------------------------------------------------------------------------------- |
| `CORTEX_SELFHEAL_COOLDOWN_S`              | `600`   | Minimum seconds between repair triggers                                               |
| `CORTEX_SELFHEAL_BACKOFF_FACTOR`          | `2`     | Exponential multiplier applied on repeated failures within the window                 |
| `CORTEX_SELFHEAL_MAX_ATTEMPTS`            | `3`     | Attempt ceiling per window before pause-and-ask guidance                              |
| `CORTEX_SELFHEAL_ATTEMPT_WINDOW_S`        | `86400` | Sliding window for counting attempts                                                  |
| `CORTEX_SELFHEAL_LOCK_STALE_S`            | `2700`  | Stale-lock age threshold (45 min)                                                     |
| `CORTEX_SELFHEAL_PROBE_COST_THRESHOLD_MS` | `600`   | Probe-cost budget; state-file-only fallback when `checkDenseReadiness()` exceeds this |
| `CORTEX_SELFHEAL_DISABLE`                 | `0`     | Kill switch — `1` disables all triggering (read-side warnings still work)             |

### Repair sequence (reuse, do not duplicate)

The orchestrator composes only existing automation, in the canonical narrowest-classifier-first order documented in `repo-cortex-workflow` and `repo-cortex-embeddings` (referenced here, not restated):

1. Classify: `node rag-index/validate-index.mjs --json` (corpus staleness) and `checkDenseReadiness()` (dense state).
2. Smallest safe repair for the classified fault: targeted/incremental `node rag-index/build-index.mjs --json` for stale/missing corpus paths; `npm run index:build-snapshot` when the browser snapshot is older than the corpus build; `npm run index:prewarm` when dense state is `cold` or `model-only` (the exact fix for today's incident).
3. Revalidate: `node scripts/agent-customization/gates/cortex-index.gate.mjs --auto-rebuild --json` plus `scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json`; record outcome + measured duration into the state file `history`.
4. On failure: honor backoff/ceiling; on ceiling, emit pause-and-ask guidance with the exact manual commands.

The chaining logic mirrors `scripts/agent-customization/hooks/refresh-cortex-after-write.mjs` (its `refreshSteps` array is the reference sequence), but the new orchestrator runs it headlessly with lock ownership, backoff accounting, and state-file writes. No repair logic is forked — the orchestrator shells out to the same npm scripts / node entry points.

### Model-facing guidance (exact templates)

Every degraded/unavailable response gains a structured `self_heal` field (camelCase JSON) plus a one-paragraph human summary prepended to the text content so harnesses that surface only text still show the warning. Templates (placeholders filled from state/lock data):

**T1 — degraded, repair just started**

```text
Cortex dense search is degraded (<dense_reason>; state=<dense_state>). Background self-heal has been STARTED: <repair_sequence> (estimated ~<est_min> min for <chunk_count> chunks; estimate based on <basis=history|chunk-count heuristic>). BM25-only results are being returned below — continue with reduced recall or fall back to native tools (grep/glob/view) and re-try Cortex after the repair completes. Do NOT re-trigger repair: a cooldown of <cooldown_min> min applies (attempt <i> of <K> in the current window). If automatic repair fails <K> times, pause and ask the user to run `npm run index:session-start` followed by `npm run index:prewarm`, then re-run the Cortex gate.
```

**T2 — degraded, repair already in flight**

```text
Cortex dense search is degraded (<dense_reason>) and self-heal is ALREADY RUNNING (started <elapsed> min ago, attempt <i>/<K>, est. remaining ~<rem_min> min). BM25-only results are being returned — continue with reduced recall or fall back to native tools. No re-trigger needed; results recover automatically when repair completes. Wait and re-try later rather than spawning new repair work.
```

**T3 — degraded, attempts exhausted (hard pause)**

```text
Cortex dense search is degraded (<dense_reason>) and automatic self-heal has FAILED <K> times in the last <window_h> h (last failure: <short_error>). Auto-repair is paused to avoid a blocker loop. PAUSE your work on RAG-dependent steps and ask the user to run `npm run index:session-start`, then `npm run index:prewarm`, then `node scripts/agent-customization/gates/cortex-index.gate.mjs --auto-rebuild --json`. BM25-only results are being returned meanwhile.
```

**T4 — server unavailable (facade spawn failure / no results at all)**

```text
The Cortex MCP server failed to start (<error_summary>). RAG search is UNAVAILABLE, not merely degraded — BM25 fallback is impossible because the server process is dead. Use native tools (grep/glob/view) for this task. Ask the user to verify TURSO_DATABASE_URL in `.vscode/mcp.json` and run `node scripts/mcp-semantic/repo-cortex-mcp.mjs` once to inspect the startup error; manual recovery commands: `npm run index:session-start`, then `npm run index:prewarm`.
```

The structured block shape (all fields always present, `null` where not applicable): `self_heal: { state, reason, action: 'started'|'in_flight'|'exhausted'|'unavailable'|'disabled', attempt, max_attempts, cooldown_s, next_allowed_at, est_duration_min, manual_recovery: [commands], guidance }` where `guidance` is the rendered paragraph above.

### Red-test strategy (applies to every implementing slice)

- Tests are `*.test.mjs` colocated suites in the jest projects `agent-customization-mjs` (guard/orchestrator/facade) and `mcp-semantic-mjs` (search tools, server).
- External effects are deterministic seams: injected probe (`readinessProbe`, following `dense-readiness.gate.mjs`'s injection pattern), injected clock (`now()`), injected spawner, injected state-dir pointing at a `rag-index/__test_tmp_session/`-style temp dir. The existing `DENSE_FORCE_STATE` env override in `dense-readiness.mjs` is reused to simulate `cold`/`model-only`/`warm` without building embeddings.
- Red expectations are asserted against the contracts in this plan (state schema, lock semantics, response `self_heal` block, template renderings, cooldown/backoff math) before any implementation exists.

## Non-goals

1. Automatically re-registering or re-enabling the manual-only heavy hooks. Registration policy changes are a separate decision.
2. Turning `pre-dispatch-freshness-hook.mjs` into a dense-health checker. It stays index-staleness-only; the existing session-start path remains untouched except as a future read-only consumer of the state file.
3. Changing gate behavior of `cortex-index.gate.mjs` (already supports `--auto-rebuild`), `dense-readiness.gate.mjs`, or `cortex-mcp-smoke.mjs`; the orchestrator invokes them as-is.
4. Any user-intervention requirement on the happy path. Waiting is acceptable; the model is told why, for how long, and what to do if repair exhausts.

## Risks and mitigations

| Risk                                                                                                                                 | Mitigation                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Repair storm: burst of degraded searches spawns parallel prewarms                                                                    | Atomic lock create (`'wx'`) + cooldown check before spawn; only the lock winner spawns                                                                                                               |
| Running server keeps reporting degraded after repair completes (process-lifetime `cachedDenseReadiness` memo in `search-corpus.mjs`) | Re-probe when degraded: memo is honored while `warm`; a degraded result is re-verified against the state file's `last_probe`/`in_flight` and re-probed at most once per `CORTEX_SELFHEAL_COOLDOWN_S` |
| Lock wedged by a crashed orchestrator                                                                                                | Heartbeat + pid-liveness + age-based stale reclamation; reclamation is itself lock-protected                                                                                                         |
| Mechanism becomes a periodic-updater-in-disguise                                                                                     | Trigger conditions are signal-based only (degraded fields, spawn failure); no timer, no interval, no sweep                                                                                           |
| Harness never fires hooks for MCP tools (kills Option A)                                                                             | Core mechanism is server-internal; Step 02 measures hook support and only then may an optional context-injection layer be added                                                                      |
| Per-call probe cost regresses search latency                                                                                         | Healthy state is memoized per process exactly as today; guard work on the degraded path is a JSON-file read + stat, never `validate-index`                                                           |
| Model ignores guidance text                                                                                                          | Guidance is duplicated into the structured `self_heal` field with explicit `action` + `manual_recovery` so agents can branch programmatically                                                        |
| New state JSON accidentally committed                                                                                                | `.gitignore` patterns added in Phase 2 (same slice as the orchestrator)                                                                                                                              |
| Backoff values wrong for real prewarm cost                                                                                           | Step 02 measures the current background prewarm duration + any logged prior runs; defaults are env-tunable                                                                                           |

## Acceptance criteria

```text
AC-001
  id: AC-001
  text: When checkDenseReadiness reports model-only or cold, a search_corpus/search_context call returns BM25 results with dense_degraded plus a populated self_heal block whose action is 'started' or 'in_flight', and exactly one repair process holds the lock.
  files: scripts/agent-customization/cortex/cortex-health-guard.mjs, scripts/mcp-semantic/tools/search-corpus.mjs, scripts/mcp-semantic/tools/search-context.mjs
  validation: npm run jest:mjs -- --testPathPatterns=cortex-health-guard --runInBand --no-coverage; npm run jest:mjs -- --testPathPatterns=search-corpus --runInBand --no-coverage
  red_test: scripts/agent-customization/cortex/cortex-health-guard.test.mjs asserts trigger-once semantics and response augmentation

AC-002
  id: AC-002
  text: Cooldown, exponential backoff, and the max-attempts ceiling behave exactly as specified: second degraded call inside cooldown never spawns; after CORTEX_SELFHEAL_MAX_ATTEMPTS failures the guidance switches to T3 pause-and-ask with the exact manual commands.
  files: scripts/agent-customization/cortex/cortex-health-guard.mjs, scripts/agent-customization/cortex/cortex-health-guard.test.mjs
  validation: npm run jest:mjs -- --testPathPatterns=cortex-health-guard --runInBand --no-coverage
  red_test: injected-clock tests for cooldown boundary, backoff doubling, attempt ceiling, window reset

AC-003
  id: AC-003
  text: The repair orchestrator holds rag-index/data/cortex-self-heal.repair.lock for its whole run, refreshes heartbeat, releases on exit (success or failure), and a crashed run's lock is reclaimed via staleness rules exactly once.
  files: scripts/agent-customization/cortex/cortex-self-heal.mjs, scripts/agent-customization/cortex/cortex-self-heal.test.mjs
  validation: npm run jest:mjs -- --testPathPatterns=cortex-self-heal --runInBand --no-coverage; node scripts/agent-customization/cortex/cortex-self-heal.mjs --dry-run --json
  red_test: orchestrator tests assert lock create/heartbeat/release/reclaim and dry-run output shape

AC-004
  id: AC-004
  text: Repair composes existing automation only — orchestrator invocations reference rag-index/validate-index.mjs, rag-index/build-index.mjs, npm run index:build-snapshot, npm run index:prewarm, scripts/agent-customization/gates/cortex-index.gate.mjs, scripts/agent-customization/gates/cortex-mcp-smoke.mjs — with no forked repair logic.
  files: scripts/agent-customization/cortex/cortex-self-heal.mjs
  validation: node scripts/agent-customization/cortex/cortex-self-heal.mjs --dry-run --json
  red_test: dry-run report lists the composed commands in classifier-first order

AC-005
  id: AC-005
  text: After a successful repair, an already-running server process observes the warm state without restart (readiness memo invalidation), and search responses drop the self_heal degraded warning.
  files: scripts/mcp-semantic/tools/search-corpus.mjs, scripts/mcp-semantic/repo-cortex-mcp.mjs
  validation: npm run jest:mjs -- --testPathPatterns=mcp-semantic --runInBand --no-coverage
  red_test: memo-invalidation test with injected probe transitioning model-only -> warm

AC-006
  id: AC-006
  text: If the facade cannot spawn the real server, the failing tools/call response includes the T4 model-facing instructions with manual recovery commands, improving on today's one-line fallbackHint; the facade still never hangs or swallows the error.
  files: scripts/agent-customization/mcp/lazy-facade-core.mjs
  validation: npm run jest:mjs -- --testPathPatterns=lazy-facade --runInBand --no-coverage
  red_test: facade test with failing spawn asserts T4 guidance and error propagation

AC-007
  id: AC-007
  text: Health on the happy path is free: when checkDenseReadiness is warm, zero state-file reads, zero lock operations, and zero spawn attempts occur (verified by counter-instrumented test).
  files: scripts/agent-customization/cortex/cortex-health-guard.mjs
  validation: npm run jest:mjs -- --testPathPatterns=cortex-health-guard --runInBand --no-coverage
  red_test: warm-path instrumentation test asserts no fs/spawn calls

AC-008
  id: AC-008
  text: All new/changed behavior is covered by the jest suites named per slice and the consolidated slice-advancement gate passes for each slice id with green evidence recorded in this plan.
  files: plans/rag-self-heal-hooks.plans.md
  validation: node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=<slice> --changed-files=<files>
  red_test: not applicable (process criterion)
```

## Open assumptions and NEEDS CLARIFICATION

1. **NEEDS CLARIFICATION (verified empirically in Phase 1 Step 02, not by assumption):** does this harness fire PreToolUse/PostToolUse hooks for MCP-routed tool names (e.g. `cortex-cortex`), and does PostToolUse deliver the tool result for inspection? Determines whether an optional Option-A context-injection layer is even possible. The recommended mechanism does not depend on the answer.

> User: Analize with the help of 3 different specialists (GLM 5.3) to agree on the optimal solution.
> **Answer:** `PostToolUse` hooks fire for native tools and for successful MCP tool calls (e.g. `neataptic-gate-mcp-list_gates`), delivering `tool_name`, `tool_input`, and `tool_result` on stdin. However, new hook registrations in `.github/hooks/cortex-refresh.json` are not picked up mid-session, and failing `cortex-cortex` calls skip `PostToolUse` entirely. Therefore an optional hook-based guidance layer is possible but cannot be the primary recovery path. The three mechanism specialists and one performance specialist all converged on the existing recommendation: Option B (server-internal core) + Option C's shared state file.

2. **NEEDS CLARIFICATION (default values proposed, sign-off requested):** cooldown 600 s, backoff ×2, max 3 attempts per 24 h, stale-lock 45 min. If the user wants different latency/safety tradeoffs, these are env-tunable without code changes.

> User: Sounds good, double check with 1 independent GLM 5.3 specilist just in case. Choose optimal solution.
> **Answer:** Confirmed by the independent performance specialist. All defaults remain as proposed.

3. **NEEDS CLARIFICATION (probe-cost budget):** `checkDenseReadiness()` on the degraded path is assumed cheap (file stats + one sqlite count). Step 02 measures it; if it exceeds ~100 ms on this machine, the guard degrades to state-file-only checks between repairs.

> User: We don't want to use a threshold that end ups always degrading pretty much by default, analize this machine, so it fires under it's reality and the degradation has a value that would trigger only during exceptional scenarios, like the machine is too slow due to some other tasks (like tests running)
> **Answer:** Measured 21 samples while the index is `model-only`: min 173 ms, p50 190 ms, p95 215 ms, max 220 ms. A threshold of 600 ms is ~3× p50 and ~2.8× p95, so it only triggers the state-file-only fallback under exceptional contention (e.g. `SQLITE_BUSY`, heavy tests). Added to the env-knob table as `CORTEX_SELFHEAL_PROBE_COST_THRESHOLD_MS = 600`.

## Implementation phases

### Phase 1 — Planning and research confirmation [DONE]

```yaml
phase: 1
title: Planning and research confirmation
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_phase: 'Phase 2 — Repair scheduler state and self-heal orchestrator'
skills:
  - 'plan-registration'
  - 'phase-handoff-workflow'
  - 'plan-sync-validation'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/rag-self-heal-hooks.plans.md --json'
acceptance_criteria:
  - 'Step 01 authors and registers this plan and passes the consolidated slice-advancement self-check'
  - 'Step 02 produces plans/rag-self-heal-hooks.research.md with empirical answers to the open assumptions'
  - 'Step 03 records green-light: true from an independent verification instance before any Phase 2 step activates'
placeholder_steps:
  - step: 1
    title: 'Author and register the self-heal plan (this pass)'
    goal: planning
    status: '[DONE]'
  - step: 2
    title: 'Research confirmation probes (hook feasibility, probe cost, duration calibration)'
    goal: researching
    status: '[DONE]'
  - step: 3
    title: 'Independent plan verification and green-light'
    goal: planning
    status: '[DONE]'
```

[DONE] Phase 1: Planning and research confirmation — plan authored and registered, research confirmed Option B + C (hooks demoted to additive-only; probe cost p50 190 ms / p95 215 ms), independent verification recorded green-light: true (2026-09-04). Original step packets, probe checklist, and raw evidence compressed to plans/rag-self-heal-hooks.logs.md.

#### Step 01 — Author and register the self-heal plan (this pass) [DONE]

[DONE] Step 01: Plan authored and registered (plans/README.md trigger phrases + plans/Roadmap.md standalone lane); slice-advancement PASS 4/4 sub-gates (P1-S01, TRIVIAL, plans-only); validate-plan-sync ok (0 errors, 0 warnings). Original packet + evidence: plans/rag-self-heal-hooks.logs.md (Phase 1, Step 01).

#### Step 02 — Research confirmation probes [DONE]

[DONE] Step 02: Research confirmed Option B + C with hooks demoted to additive-only (PostToolUse fires for native tools and successful MCP calls; skipped on MCP-call errors; frozen mid-session); probe cost p50 190 ms / p95 215 ms (21 samples); env knobs confirmed (cooldown 600 s, backoff x2, max 3 / 24 h, stale-lock 2700 s, probe-cost 600 ms); prewarm duration for ~25k chunks honestly partial. Artifact: plans/rag-self-heal-hooks.research.md. Original packet + probe checklist + evidence: plans/rag-self-heal-hooks.logs.md (Phase 1, Step 02).

#### Step 03 — Independent plan verification and green-light [DONE]

[DONE] Step 03: Fresh independent 01-planning verification instance validated packets, AC traceability (AC-001..AC-008, 100% coverage), scope honesty, and research answers; recorded green-light: true (2026-09-04); slice-advancement PASS 4/4 sub-gates (P1-S03, TRIVIAL). Original packet + evidence: plans/rag-self-heal-hooks.logs.md (Phase 1, Step 03).

### Phase 2 — Repair scheduler state and self-heal orchestrator [DONE]

```yaml
phase: 2
title: 'Repair scheduler state and self-heal orchestrator'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (phase packet)'
copy_paste: ready
next_phase: 'Phase 3 — Server-integrated detection and model-facing guidance'
skills:
  - 'phase-handoff-workflow'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P2-S02 --changed-files=scripts/agent-customization/cortex/cortex-health-guard.mjs,scripts/agent-customization/cortex/cortex-self-heal.mjs'
acceptance_criteria:
  - 'AC-001 guard/scheduler portion, AC-002, AC-003, AC-004 satisfied with green evidence'
placeholder_steps:
  - step: 1
    title: 'Phase kickoff (re-read plan, confirm research answers absorbed)'
    goal: planning
    status: '[DONE]'
  - step: 2
    title: 'Guard module + orchestrator (red-green, 4 slices)'
    goal: implementing
    status: '[DONE]'
```

#### Step 01 — Phase kickoff (re-read plan, confirm research answers absorbed) [DONE]

[DONE] Step 01: Kickoff confirmed all three research answers absorbed by Phase 2 (hooks demoted to additive-only with zero hook registrations in Phase 2 slices; all seven env-knob defaults present; probe-cost threshold 600 ms ≈ 3x measured p50 190 ms / p95 215 ms); two absorption gaps patched into the Step 02 packet before activation (probe-cost-threshold, kill-switch, and pid-liveness-reclaim coverage). slice-advancement PASS 4/4 sub-gates (P2-S01, TRIVIAL, plans-only). Full kickoff record: plans/rag-self-heal-hooks.logs.md (Phase 2, Step 01).

#### Step 02 — Guard module and self-heal orchestrator (TDD) [DONE]

[DONE] Step 02: Four-slice red-green TDD pass delivered scripts/agent-customization/cortex/cortex-health-guard.mjs (evaluateSelfHeal({probe, now, stateDir, spawner}) decision engine: state schema, cooldown/backoff/ceiling, lock management, kill switch, probe-cost fallback at 600 ms ≈ 3x measured p50, zero-cost warm path) and scripts/agent-customization/cortex/cortex-self-heal.mjs (runRepair/createLock/refreshLock/releaseLock/reclaimStaleLock/dryRunReport + dry-run CLI; .gitignore for state/lock artifacts). Both suites 15/15 PASS under npm run jest:mjs; dry-run composes the 6 existing-automation commands classifier-first with zero --force; convergence-tracker PASS; slice-advancement PASS 4/4 (P2-S02-A, TRIVIAL) and PASS 7/7 (P2-S02-C/D, FULL incl. shared-validation, code-coverage, specialist-review). AC-002/003/004/007 green. Original step packet with all four slice definitions + raw evidence: plans/rag-self-heal-hooks.logs.md (Phase 2, Step 02).

### Phase 3 — Server-integrated detection and model-facing guidance [DONE]

```yaml
phase: 3
title: 'Server-integrated detection and model-facing guidance'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (phase packet)'
copy_paste: ready
next_phase: 'Phase 4 — Server-unavailable guidance seam (facade)'
skills:
  - 'phase-handoff-workflow'
  - 'repo-cortex-embeddings'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P3-S02 --changed-files=scripts/mcp-semantic/tools/search-corpus.mjs,scripts/mcp-semantic/tools/search-corpus.test.mjs,scripts/mcp-semantic/tools/search-context.mjs,scripts/mcp-semantic/tools/search-context.test.mjs,scripts/mcp-semantic/tools/index-stats.mjs,scripts/mcp-semantic/tools/index-stats.test.mjs,scripts/mcp-semantic/repo-cortex-mcp.mjs,scripts/mcp-semantic/repo-cortex-mcp.test.mjs'
acceptance_criteria:
  - 'AC-001 response augmentation and trigger-once wiring'
  - 'AC-005 post-repair warm observation without server restart'
placeholder_steps:
  - step: 1
    title: 'Phase kickoff'
    goal: planning
    status: '[DONE]'
  - step: 2
    title: 'Search-tool and index_stats wiring (red-green, 4 slices)'
    goal: implementing
    status: '[DONE]'
```

[DONE] Phase 3: Server-integrated detection and model-facing guidance — four-slice red-green TDD pass wired `evaluateSelfHeal` into search-corpus.mjs, search-context.mjs, index-stats.mjs, and repo-cortex-mcp.mjs; 349/349 mjs tests green across four suites; live DENSE_FORCE_STATE=model-only smoke produced a structured self_heal block with T3 exhausted guidance and BM25 fallback; slice-advancement PASS 7/7 for P3-S02; docs-scout README/JSDoc drift fixed. Original step packets + raw evidence: plans/rag-self-heal-hooks.logs.md (Phase 3).

#### Step 01 — Phase kickoff (server-integration absorption) [DONE]

[DONE] Step 01: Research and Phase 2 module absorption confirmed; gaps patched into P3-S02 packet before activation. Evidence: plans/rag-self-heal-hooks.logs.md (Phase 3, Step 01).

#### Step 02 — Search-tool and index_stats wiring (TDD) [DONE]

[DONE] Step 02: Four-slice red-green TDD pass delivered search-tool/index_stats wiring and docs-quality fixes. Evidence: plans/rag-self-heal-hooks.logs.md (Phase 3, Step 02).

### Phase 4 — Server-unavailable guidance seam (facade) [DONE]

```yaml
phase: 4
title: 'Server-unavailable guidance seam (facade)'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (phase packet)'
copy_paste: ready
next_phase: 'Phase 5 — Documentation and closure'
skills:
  - 'phase-handoff-workflow'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P4-S02 --changed-files=scripts/agent-customization/mcp/lazy-facade-core.mjs,scripts/agent-customization/mcp/lazy-facade-core.test.mjs,plans/rag-self-heal-hooks.plans.md'
acceptance_criteria:
  - 'AC-006 T4 guidance on spawn failure'
placeholder_steps:
  - step: 1
    title: 'Phase kickoff'
    goal: planning
    status: '[DONE]'
  - step: 2
    title: 'Facade failure guidance (red-green, 3 slices)'
    goal: implementing
    status: '[DONE]'
```

[DONE] Phase 4: Server-unavailable guidance seam — three-slice red-green TDD pass delivered T4 spawn-failure guidance in `lazy-facade-core.mjs` and focused contract tests; 71/71 lazy-facade tests green, 82/82 combined facade suites green, 100% branch coverage, slice-advancement PASS 7/7 (FULL). Original step packets + raw evidence: plans/rag-self-heal-hooks.logs.md (Phase 4).

#### Step 01 — Phase kickoff (facade failure guidance absorption) [DONE]

[DONE] Step 01: Phase kickoff confirmed Phase 3 evidence intact and P4-S02 packet consistent; no absorption gaps required. Evidence: plans/rag-self-heal-hooks.logs.md (Phase 4, Step 01).

#### Step 02 — Facade failure guidance (TDD) [DONE]

[DONE] Step 02: Three-slice red-green TDD pass delivered T4 facade guidance and green validation. Evidence: plans/rag-self-heal-hooks.logs.md (Phase 4, Step 02).

### Phase 5 — Documentation and closure [DONE]

```yaml
phase: 5
title: 'Documentation and closure'
status: '[WIP]'
goal: planning
expansion: steps
auto_expand: false
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (phase packet)'
copy_paste: ready
next_phase: 'plan archive'
skills:
  - 'phase-handoff-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/rag-self-heal-hooks.plans.md --json'
acceptance_criteria:
  - 'Skills reference the self-heal surface; plan closed with PlanUpdate and compressed history'
placeholder_steps:
  - step: 1
    title: 'Documentation alignment (documenting, expansion none)'
    goal: documenting
    status: '[PLANNED]'
  - step: 2
    title: 'Compression and closure (logging, expansion none)'
    goal: logging
    status: '[PLANNED]'
```

#### Step 01 — Documentation alignment [DONE]

[DONE] Step 01: Added self-heal surface pointers to `repo-cortex-workflow` and `repo-cortex-embeddings` skills without duplicating repair sequences. Evidence: `plans/rag-self-heal-hooks.logs.md` (Phase 5, Step 01).

#### Step 02 — Compression and closure [DONE]

[DONE] Step 02: Plan and Phase 5 compressed; README/Roadmap flipped to `[DONE]`; tracker pair archived to `plans/completed/`. Evidence: `plans/rag-self-heal-hooks.logs.md` (Phase 5, Step 02).

## Validation gates

Every slice and step must pass the consolidated Tier-1 gate before it is marked [DONE]. The sub-gates it consolidates (plan-sync, step-packet, plan-slice-quality, plan-command-lint) are never run individually as validation.

- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=<slice-id> --changed-files=<changed-files>`
- `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/rag-self-heal-hooks.plans.md --json`

## Latest validation evidence

```text
Phase 4 Step 01 kickoff (2026-09-05):
- Phase 3 [DONE] compression and evidence preserved; P4-S02 packet reviewed for absorption gaps — no patches required.
- Phase 4 status flipped [WIP]; Step 01 [DONE]; Step 02 [WIP].
- slice-advancement PASS 4/4 sub-gates (P4-S01, TRIVIAL, plans-only): plan-sync, step-packet, plan-slice-quality, plan-command-lint.
- README/Roadmap already register plan [WIP], Phases 1-3 [DONE], Phase 4 next; no index edits required.
```

```text
Phase 4 Step 02 P4-S02-A red (2026-09-05):
- RED confirmed for the right reason in scripts/agent-customization/mcp/lazy-facade-core.test.mjs: 6 new focused its inside describe('createLazyFacade') covering the T4 facade contract — T4 guidance markers ('Cortex MCP server failed to start', 'RAG search is UNAVAILABLE', 'grep/glob/view'); manual-recovery markers ('TURSO_DATABASE_URL', 'npm run index:session-start', 'npm run index:prewarm', 'node scripts/mcp-semantic/repo-cortex-mcp.mjs'); error summary 'spawn failed' embedded in guidance; self_heal action 'unavailable' with full 10-field schema and manual_recovery subset; fallbackHint removed; error propagation + no spawn retry across two dispatches.
- Focused validation (allow-listed) `npm run jest:mjs -- --testPathPatterns=lazy-facade --runInBand --no-coverage`: exit code 1 — 5 failed / 65 passed / 70 total. All 5 failures are missing-implementation assertions (4x TypeError reading 'includes'/'action' of undefined payload.guidance/payload.self_heal; 1x fallbackHint still present: "Start the real cortex server manually or check the spawn command."). The no-retry/propagation it passes as a regression guard over already-correct behavior (cached rejected initPromise).
- Fixture: existing mock seam — childProcess.spawn throws Error('spawn failed'), fs.readFileSync returns router snapshot with tool name 'cortex', new cortexConfig (target 'cortex', spawn command scripts/mcp-semantic/repo-cortex-mcp.mjs); makeRouterSnapshot extended to accept a router tool name (backward-compatible default); beforeEach mock resets unchanged; no timers/files.
- slice-advancement PASS 4/4 sub-gates (P4-S02-A, TRIVIAL) via run_gate_check: plan-sync, step-packet, plan-slice-quality, plan-command-lint.
- Delegation: tests authored by unit-test-writer (Tier 3); contract-fidelity corrections applied by 03-red-testing orchestrator review (added 'grep/glob/view' + 'TURSO_DATABASE_URL' markers, added error-summary it, relaxed manual_recovery deep-equal to subset assertion per designed contract).
- Slice status intentionally left [PLANNED] — orchestrator owns transitions; P4-S02-C records green evidence.
- Expected green (P4-S02-B): spawn-failure payload renders T4 template guidance with <error_summary> filled by the spawn failure message plus a full self_heal block (action 'unavailable', manual_recovery containing both npm commands), fallbackHint removed, error still propagates, spawn called exactly once.
```

```text
Phase 4 Step 02 P4-S02-B implementing (2026-09-05):
- Implemented T4 spawn-failure guidance in scripts/agent-customization/mcp/lazy-facade-core.mjs: `spawnCommand` propagated through dispatch/serverDispatch context; new `isSpawnFailureError()` classifies spawn-time errors; new `buildCortexSpawnFailurePayload()` renders T4 guidance string, error summary, and full self_heal block; catch block emits T4 payload only for target 'cortex' + spawn failure, keeping generic fallbackHint path for all other errors and preserving JSON-RPC tool-error semantics (isError: true, content[0].text JSON).
- Added focused regression test for the 'Spawn returned an invalid child process.' branch to reach 100% branch coverage on lazy-facade-core.mjs.
- Targeted tests: `npm run jest:mjs -- --testPathPatterns=lazy-facade --runInBand --no-coverage` → 71 passed / 71 total; combined lazy-facade + cortex-facade + devtools-facade → 82 passed / 82 total.
- Type/lint preflight: `npx tsc --noEmit -p tsconfig.json` PASS; `npm run quality:folder -- --folder=scripts/agent-customization/mcp` PASS (0 ESLint errors across 5 files).
- Coverage: `npm run jest:mjs -- --runInBand --coverage --testPathPatterns=lazy-facade` + `node scripts/agent-customization/gates/merge-coverage-summaries.mjs` produced 100% lines/statements/functions/branches for scripts/agent-customization/mcp/lazy-facade-core.mjs.
- slice-advancement PASS 7/7 sub-gates (P4-S02, FULL): plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review.
- Slice status flipped [DONE]; P4-S02-C set [WIP] for green-testing handoff.
```

```text
Phase 4 Step 02 P4-S02-C green-testing (2026-09-05):
- Allow-listed focused validation `npm run jest:mjs -- --testPathPatterns=lazy-facade --runInBand --no-coverage`: 71 passed / 71 total.
- Regression guard across related facade surfaces `npm run jest:mjs -- --testPathPatterns='lazy-facade|cortex-facade|devtools-facade' --runInBand --no-coverage`: 82 passed / 82 total.
- Re-ran consolidated `slice-advancement` gate for P4-S02 with changed files (lazy-facade-core.mjs, lazy-facade-core.test.mjs, plans/rag-self-heal-hooks.plans.md): PASS 7/7 sub-gates (plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review), severity FULL.
- No new source changes in this slice; coverage and specialist review evidence carried forward from P4-S02-B and confirmed passing by the consolidated gate.
- Slice status flipped [DONE].
```

```text
Phase 1 evidence compressed 2026-09-04 (07-logging phase compression) — full raw evidence moved to plans/rag-self-heal-hooks.logs.md:
- P1-S01 (authoring + registration): PASS slice-advancement 4/4 sub-gates (TRIVIAL, plans-only); validate-plan-sync ok: true (0 errors, 0 warnings); registered in plans/README.md and plans/Roadmap.md.
- P1-S02 (research): PASS slice-advancement 4/4 sub-gates; artifact plans/rag-self-heal-hooks.research.md; Option B + C confirmed with hooks demoted to additive-only; probe cost p50 190 ms / p95 215 ms (21 samples); env knobs confirmed.
- P1-S03 (verification): PASS slice-advancement 4/4 sub-gates (TRIVIAL); green-light: true recorded by a fresh independent 01-planning instance (2026-09-04); AC coverage 100% (AC-001..AC-008); scope honesty verified.
- Known partial (non-blocking): full end-to-end prewarm duration for ~25k chunks not collected; production repairs record real durations in the state-file history.
- Transition note: step flips on this plan are explicit edits (workflow-update-sync matcher hazards recorded as learning events). Phase 1 history compressed per the phase-compression rule before Phase 2 activation; between-phases snapshot resolves activeStep = null (expansion: steps + auto_expand: false).
```

```text
Phase 2 evidence compressed 2026-09-04 (07-logging phase compression) — full raw evidence moved to plans/rag-self-heal-hooks.logs.md:
- P2-S01 (Phase kickoff): PASS slice-advancement 4/4 sub-gates (TRIVIAL, plans-only); research absorption confirmed (probe-cost threshold 600 ms ≈ 3x measured p50 190 ms / p95 215 ms; all 7 env knobs present; hooks additive-only, zero registrations in Phase 2 slices).
- P2-S02-A (red): RED confirmed for the right reason (missing modules, not syntax/fixture errors); plan-exact evaluateSelfHeal contract + snake_case state/lock schemas encoded in both colocated suites; a drifted first writer pass was deleted before commit; slice-advancement PASS 4/4 (TRIVIAL) after converting 10 inline dependency lists to block-style (parser quote-strip defect).
- P2-S02-B (implementing): cortex-health-guard.mjs — evaluateSelfHeal API, state schema, cooldown/backoff/ceiling math, lock management, kill switch, probe-cost fallback, zero-cost warm path; guard suite 15/15 PASS under jest:mjs.
- P2-S02-C (implementing): cortex-self-heal.mjs (runRepair/createLock/refreshLock/releaseLock/reclaimStaleLock/dryRunReport + CLI) + .gitignore for state/lock artifacts; two red-fixture repairs disclosed (missing fsCounts Proxy block; dead-pid live-holder fixture); 15/15 PASS; dry-run composes 6 existing-automation commands classifier-first with zero --force; slice-advancement PASS 7/7 (FULL: plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review).
- P2-S02-D (green): both suites 15/15 PASS under jest:mjs; dry-run JSON recorded (6 commands, ordering classifier-first, no lock taken); convergence-tracker PASS; slice-advancement PASS 7/7 (FULL; four Step 02 source/test files). Step 02 [DONE]; Phase 2 compressed [DONE] (2026-09-04).
- Gate note: the slice-advancement gate MCP wrapper requires equals-form args (--slice-id=..., --changed-files=...); plain-JSON args are rejected (learning event recorded at P2-S02-C).
```

```text
Phase 3 evidence compressed 2026-09-04 (07-logging phase compression) — full raw evidence moved to plans/rag-self-heal-hooks.logs.md:
- P3-S01 (Phase kickoff): PASS slice-advancement 4/4 sub-gates (TRIVIAL, plans-only); validate-plan-sync ok: true (0 errors, 0 warnings); research artifact absorbed; gaps patched into P3-S02 packet.
- P3-S02-A (red): RED confirmed for the right reason across four colocated test suites (missing seams); slice-advancement PASS 4/4 sub-gates.
- P3-S02-B (implementing): search-corpus.mjs wired to evaluateSelfHeal, invalidateDenseReadinessCache seam exposed; 154 tests PASS; tsc + lint PASS.
- P3-S02-C (implementing): search-context.mjs, index-stats.mjs, repo-cortex-mcp.mjs wired; 349/349 tests PASS across four suites; all changed source files 100% coverage.
- P3-S02-D (green): live DENSE_FORCE_STATE=model-only smoke produced structured self_heal block with T3 exhausted guidance + BM25 fallback; slice-advancement PASS 7/7 for P3-S02; docs-scout README/JSDoc drift fixed.
- Step 02 [DONE]; Phase 3 compressed [DONE] (2026-09-04).
```

## Latest validation evidence

```text
Phase 5 documentation alignment (2026-09-05):
- Added "Self-heal surface" pointer section to .github/skills/repo-cortex-workflow/SKILL.md.
- Added "Degraded-state response" section to .github/skills/repo-cortex-embeddings/SKILL.md linking to the self_heal response contract.
- No repair sequences duplicated; canonical sequences remain in cortex-self-heal.mjs / cortex-health-guard.mjs.
- slice-advancement PASS 4/4 sub-gates (P5-S01, TRIVIAL).
- validate-plan-sync PASS 0 errors, 0 warnings for plans/rag-self-heal-hooks.plans.md.
- cortex-index gate reports index_fresh=true, skill family fresh=true, auto_rebuild_success=true; workflow_mcp_alive=false is a live host-side MCP server state issue, not a documentation gap.
- Phase 5 status flipped [DONE]; Step 01 [DONE]; Step 02 [DONE]; plan closed and archived.
```

```text
Phase 4 compression and closure (2026-09-05):
- Phase 4 status flipped [DONE]; Step 01 [DONE]; Step 02 [DONE]; all three slices P4-S02-A/B/C [DONE].
- Phase 4 detailed step packets and raw validation evidence moved to plans/rag-self-heal-hooks.logs.md.
- Plan file now carries compact [DONE] coverage notes per the phase-compression rule.
- slice-advancement PASS 7/7 sub-gates (P4-S02, FULL) confirmed before compression.
- Active frontier moved to Phase 5 — Documentation and closure.
```

## Final state

- All phases (1–5) completed and compressed.
- Source/test artifacts delivered:
  - `scripts/agent-customization/cortex/cortex-health-guard.mjs` + `.test.mjs`
  - `scripts/agent-customization/cortex/cortex-self-heal.mjs` + `.test.mjs`
  - `scripts/mcp-semantic/tools/search-corpus.mjs` + `.test.mjs`
  - `scripts/mcp-semantic/tools/search-context.mjs` + `.test.mjs`
  - `scripts/mcp-semantic/tools/index-stats.mjs` + `.test.mjs`
  - `scripts/mcp-semantic/repo-cortex-mcp.mjs` + `.test.mjs`
  - `scripts/agent-customization/mcp/lazy-facade-core.mjs` + `.test.mjs`
  - `.github/skills/repo-cortex-workflow/SKILL.md` (Self-heal surface pointer)
  - `.github/skills/repo-cortex-embeddings/SKILL.md` (Degraded-state response pointer)
- README/Roadmap entries flipped to `[DONE]` and plan/logs pair archived to `plans/completed/`.
- Latest validation: `slice-advancement` PASS for P5-S02; `validate-plan-sync` PASS; `log-completion-marker` gate PASS; `stale-wip-plans` gate PASS.

## Audit summary

- Phase 1: plan authored, Option B+C mechanism approved, green-light recorded.
- Phase 2: guard + orchestrator modules implemented with cooldown/backoff/lock/state semantics and dry-run repair composition.
- Phase 3: search tools and `index_stats` wired to emit `self_heal` blocks; README/JSDoc drift corrected.
- Phase 4: facade T4 spawn-failure guidance implemented with full test coverage.
- Phase 5: skill pointer docs added; plan compressed and archived.

## Reopen conditions

- New fault surfaces not covered by current detection signals require an active plan update and a fresh verification green-light before implementation.
- Reopen by moving the archived pair back to `plans/` or creating a successor tracker that references `plans/completed/rag-self-heal-hooks.plans.md`.

## Audit log

See `plans/rag-self-heal-hooks.logs.md` for per-phase step packets, raw validation evidence, and fix-loop records.
