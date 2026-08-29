# RAG / Cortex Self-Healing Hooks
**Status:** [WIP]

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

| Option | Summary | Verdict |
| --- | --- | --- |
| A — Environment hooks (PreToolUse/PostToolUse matching MCP tool names) | Hooks registered in `.github/hooks/cortex-refresh.json` are `{command}`-only entries; per-tool filtering happens inside the script (see `post-write-reindex-hook.mjs`). Whether this harness fires hooks for MCP-routed tool calls (`cortex-cortex`) is **unverified**; if it does not fire, the safety net silently never triggers — the exact failure class this plan exists to kill. Hooks also run on every matching call (hot-path overhead) and PostToolUse result inspection support is unconfirmed. | **Demoted** — optional additive layer only after Step 02 empirically proves MCP-tool matching + context injection/block-reason support. |
| B — Server-internal self-heal | Detection lives exactly where the `dense_degraded`/`dense_reason`/`dense_state` fields are already produced (`scripts/mcp-semantic/tools/search-corpus.mjs`, `search-context.mjs`; `index_stats` in `scripts/mcp-semantic/repo-cortex-mcp.mjs`). One shared guard module covers every consumer (facade tool, gates, future clients) with **zero registration**. The model-facing instruction block rides inside the structured response the model already parses. Async detached repair spawn follows the proven `pre-dispatch-freshness-hook.mjs` pattern. | **Recommended.** Requires care: fire-and-forget spawn gated by cooldown/lock so a burst of searches during repair never spawns N repairs; and the process-lifetime `cachedDenseReadiness` memo must be invalidated after repair or a running server keeps answering "degraded" forever. |
| C — Hybrid (server-internal + shared state + hooks for non-MCP surfaces) | The shared health-state file is genuinely valuable for cross-surface coordination (session-start hook and gates can consult it later, read-only). But hooks for MCP-call detection add nothing over B and reintroduce A's risks. | **Partially adopted** — adopt C's shared state file + consult-only readers; adopt **no** new hook registrations. |

**Recommendation: Option B with Option C's shared state file ("server-internal core, shared-state coordination").** Search-time degraded handling, cooldown/backoff bookkeeping, repair triggering, and model guidance all live in the MCP server process where the signal originates and where the response is assembled; the state/lock files under `rag-index/data/` make the behavior observable and let non-MCP surfaces (session-start hook, gates, humans) consult repair status without duplicating probes. This is the most reliable zero-intervention design: no harness-configuration dependency can silently void the safety net, guidance is guaranteed to reach the model in the tool response it is already reading, and reuse of existing repair scripts stays compile-time visible in one orchestrator module.

### Detection signal inventory (codified by the guard module)

| Source | Signal | Meaning | Probe cost |
| --- | --- | --- | --- |
| `search_corpus` / `search_context` responses | `dense_degraded: true`, `dense_reason`, `dense_state: cold|model-only` | Dense path unusable; BM25-only fallback active | Free — already computed per call via `checkDenseReadiness()` |
| `rag-index/dense-readiness.mjs` `checkDenseReadiness()` | `state: cold|model-only|warm`, `chunk_count`, `embedding_count` | Cold = ONNX model cache absent; model-only = model present, embeddings missing/incomplete (today's incident: expected 24,957, found 0) | Cheap — file-stat + sqlite row count; memoized per process (must invalidate after repair) |
| `index_stats` | `total_chunks` vs embeddings row-count mismatch | Independent confirmation of embedding-store emptiness | Cheap — same sqlite reads |
| `node rag-index/validate-index.mjs --json` | `stale_paths`, `missing_paths`, per-family `family_fresh` booleans | BM25 corpus staleness (stale-result symptom: hits from deleted files) | Heavy — **never** run synchronously on the search path; run inside the repair orchestrator only |
| `scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json` | MCP liveness | Real server process spawn/liveness probe | Runs in Phase-1 hook and gates; consulted read-only by the orchestrator after repair |
| Facade lazy-spawn failure (`scripts/agent-customization/mcp/lazy-facade-core.mjs`) | Spawn error / `tools/call` transport failure | MCP "dead / no results at all" — the only hard-block case | Free — existing error path, currently a one-line `fallbackHint` |

### State and lock files (new artifacts, following existing conventions)

Freshness state today lives in `rag-index/data/freshness-manifest.json` (single writer: `validate-index.mjs`; atomic persistence conventions). The new artifacts follow the same shape:

- `rag-index/data/cortex-self-heal-state.json` — **single writer: `cortex-self-heal.mjs` (the repair orchestrator); all other surfaces are read-only.** Written atomically (write-then-rename). Contents: `version`, `last_probe {state, dense_reason, chunk_count, embedding_count, probed_at}`, `cooldown {cooldown_s, last_trigger_at}`, `backoff {attempts_in_window, window_started_at, backoff_factor, next_allowed_at}`, `max_attempts`, `in_flight {pid, started_at, attempt, reason, est_duration_min} | null`, `history` (last 5 repair runs with duration and result, used to estimate future durations).
- `rag-index/data/cortex-self-heal.repair.lock` — mutual exclusion across every repair-capable surface. Created atomically (`fs.open` with `'wx'`) by the guard at the moment it decides to trigger repair, then owned by the detached orchestrator process. Lock record: `{pid, started_at, heartbeat_at, attempt, reason}`. Stale when `heartbeat_at` is older than `CORTEX_SELFHEAL_LOCK_STALE_S` **or** the owning pid is no longer alive; a stale lock is reclaimed by the next caller after a single forced `unlink` retry.
- `.gitignore` must be extended so both files (and `rag-index/data/*.lock`) are never committed — current ignore patterns cover `*.sqlite`, `hook-context/*.json`, and `mcp-session-override.json` but not new JSON state files.

Default knobs (env-overridable, same convention as `CORTEX_GRACE_WINDOW_S` / `CORTEX_STALENESS_THRESHOLD_S` in `pre-dispatch-freshness-hook.mjs`):

| Env var | Default | Meaning |
| --- | --- | --- |
| `CORTEX_SELFHEAL_COOLDOWN_S` | `600` | Minimum seconds between repair triggers |
| `CORTEX_SELFHEAL_BACKOFF_FACTOR` | `2` | Exponential multiplier applied on repeated failures within the window |
| `CORTEX_SELFHEAL_MAX_ATTEMPTS` | `3` | Attempt ceiling per window before pause-and-ask guidance |
| `CORTEX_SELFHEAL_ATTEMPT_WINDOW_S` | `86400` | Sliding window for counting attempts |
| `CORTEX_SELFHEAL_LOCK_STALE_S` | `2700` | Stale-lock age threshold (45 min) |
| `CORTEX_SELFHEAL_DISABLE` | `0` | Kill switch — `1` disables all triggering (read-side warnings still work) |

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

| Risk | Mitigation |
| --- | --- |
| Repair storm: burst of degraded searches spawns parallel prewarms | Atomic lock create (`'wx'`) + cooldown check before spawn; only the lock winner spawns |
| Running server keeps reporting degraded after repair completes (process-lifetime `cachedDenseReadiness` memo in `search-corpus.mjs`) | Re-probe when degraded: memo is honored while `warm`; a degraded result is re-verified against the state file's `last_probe`/`in_flight` and re-probed at most once per `CORTEX_SELFHEAL_COOLDOWN_S` |
| Lock wedged by a crashed orchestrator | Heartbeat + pid-liveness + age-based stale reclamation; reclamation is itself lock-protected |
| Mechanism becomes a periodic-updater-in-disguise | Trigger conditions are signal-based only (degraded fields, spawn failure); no timer, no interval, no sweep |
| Harness never fires hooks for MCP tools (kills Option A) | Core mechanism is server-internal; Step 02 measures hook support and only then may an optional context-injection layer be added |
| Per-call probe cost regresses search latency | Healthy state is memoized per process exactly as today; guard work on the degraded path is a JSON-file read + stat, never `validate-index` |
| Model ignores guidance text | Guidance is duplicated into the structured `self_heal` field with explicit `action` + `manual_recovery` so agents can branch programmatically |
| New state JSON accidentally committed | `.gitignore` patterns added in Phase 2 (same slice as the orchestrator) |
| Backoff values wrong for real prewarm cost | Step 02 measures the current background prewarm duration + any logged prior runs; defaults are env-tunable |

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
2. **NEEDS CLARIFICATION (default values proposed, sign-off requested):** cooldown 600 s, backoff ×2, max 3 attempts per 24 h, stale-lock 45 min. If the user wants different latency/safety tradeoffs, these are env-tunable without code changes.
3. **NEEDS CLARIFICATION (probe-cost budget):** `checkDenseReadiness()` on the degraded path is assumed cheap (file stats + one sqlite count). Step 02 measures it; if it exceeds ~100 ms on this machine, the guard degrades to state-file-only checks between repairs.

## Phase 1 — Planning and research confirmation
**Status:** [WIP]

```yaml
phase: 1
title: Planning and research confirmation
status: '[WIP]'
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
    status: '[WIP]'
  - step: 2
    title: 'Research confirmation probes (hook feasibility, probe cost, duration calibration)'
    goal: researching
    status: '[PLANNED]'
  - step: 3
    title: 'Independent plan verification and green-light'
    goal: planning
    status: '[PLANNED]'
```

### Phase 1, Step 01 — Author and register the self-heal plan (this pass)
**Status:** [WIP]

```yaml
phase: 1
step: 1
title: 'Author and register the self-heal plan (this pass)'
status: '[WIP]'
goal: planning
expansion: 'none'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Phase 1, Step 02 — Research confirmation probes'
skills:
  - 'plan-registration'
  - 'phase-handoff-workflow'
  - 'plan-sync-validation'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P1-S01 --changed-files=plans/rag-self-heal-hooks.plans.md,plans/README.md,plans/Roadmap.md'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/rag-self-heal-hooks.plans.md --json'
acceptance_criteria:
  - 'Plan file exists with mechanism decision, detection inventory, state/lock design, templates, risks, and AC block'
  - 'Plan registered in plans/README.md trigger list and plans/Roadmap.md standalone meta-workflow lane with matching [WIP] status'
  - 'Consolidated slice-advancement gate passes with plans-only changed-files (TRIVIAL slice)'
  - 'Latest validation evidence updated honestly; no green-light self-awarded'
```

Deliverable of this pass: this file plus the two registration edits. Evidence goes to `## Latest validation evidence`.

### Phase 1, Step 02 — Research confirmation probes
**Status:** [PLANNED]

```yaml
phase: 1
step: 2
title: 'Research confirmation probes (hook feasibility, probe cost, duration calibration)'
status: '[PLANNED]'
goal: researching
expansion: 'none'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Phase 1, Step 03 — Independent plan verification and green-light'
skills:
  - 'research-methodology'
  - 'repo-cortex-workflow'
  - 'repo-cortex-embeddings'
  - 'evidence-based-changes'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-S02'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P1-S02 --changed-files=plans/rag-self-heal-hooks.research.md,plans/rag-self-heal-hooks.plans.md'
acceptance_criteria:
  - 'Research artifact plans/rag-self-heal-hooks.research.md exists answering: (a) whether hooks fire for MCP tool names with result inspection, (b) measured checkDenseReadiness latency, (c) observed prewarm duration(s) for ~25k chunks from the currently running job and artifacts'
  - 'Mechanism recommendation re-confirmed or revised with one-paragraph justification recorded in the plan'
  - 'Env-knob defaults confirmed or adjusted based on measurements'
```

Probe checklist for the research dispatch (do not leave implicit):
1. Empirically determine whether PreToolUse/PostToolUse hooks observe MCP calls: add a temporary, reversible diagnostic hook entry that records `tool_name` into `rag-index/data/hook-context/`, make one `cortex-cortex` call, inspect the capture, then revert the diagnostic registration.
2. Time `node rag-index/dense-readiness.mjs --json` (and `index_stats`) on this machine; record p50/p95-style numbers.
3. Collect prewarm duration evidence: the currently running `npm run index:prewarm` background job and any duration fields in artifacts/logs.
4. Re-confirm the insertion points listed in Phase 3/4 against current sources (line-level notes in the artifact).

### Phase 1, Step 03 — Independent plan verification and green-light
**Status:** [PLANNED]

```yaml
phase: 1
step: 3
title: 'Independent plan verification and green-light'
status: '[PLANNED]'
goal: planning
expansion: 'none'
mode: verification
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet) + plans/rag-self-heal-hooks.research.md'
copy_paste: ready
next_step: 'Phase 2, Step 01 — phase kickoff'
skills:
  - 'plan-sync-validation'
  - 'phase-handoff-workflow'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P1-S03 --changed-files=plans/rag-self-heal-hooks.plans.md'
acceptance_criteria:
  - 'Fresh 01-planning verification instance validates packets, AC traceability, scope honesty, and research answers; records green-light: true with evidence in ## Latest validation evidence, or records blockers'
  - 'On blockers: verification instance reports them only — it must not self-dispatch patch cycles'
```

## Phase 2 — Repair scheduler state and self-heal orchestrator
**Status:** [PLANNED]

```yaml
phase: 2
title: 'Repair scheduler state machine and self-heal orchestrator'
status: '[PLANNED]'
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
    status: '[PLANNED]'
  - step: 2
    title: 'Guard module + orchestrator (red-green, 4 slices)'
    goal: implementing
    status: '[PLANNED]'
```

### Phase 2, Step 02 — Guard module and self-heal orchestrator (TDD)
**Status:** [PLANNED]

```yaml
phase: 2
step: 2
title: 'Guard module and self-heal orchestrator (TDD)'
status: '[PLANNED]'
goal: implementing
expansion: 'slices'
auto_expand: true
tdd_sequence: 'red-green'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Phase 3, Step 01 — phase kickoff'
skills:
  - 'red-testing'
  - 'implementation-standards'
  - 'testing'
  - 'green-testing'
  - 'repo-cortex-workflow'
specialists:
  - 'unit-test-writer'
  - 'property-based-test-writer'
  - 'implementation-executor'
constitution_check: 'pass — two new small modules composing existing scripts; no new subsystem or periodic scheduler'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P2-S02-A'
validation:
  - 'npm run jest:mjs -- --testPathPatterns=cortex-health-guard --runInBand --no-coverage'
  - 'npm run jest:mjs -- --testPathPatterns=cortex-self-heal --runInBand --no-coverage'
  - 'node scripts/agent-customization/cortex/cortex-self-heal.mjs --dry-run --json'
acceptance_criteria:
  - 'AC-002 cooldown/backoff/ceiling'
  - 'AC-003 lock lifecycle and stale reclamation'
  - 'AC-004 composed-command dry-run in classifier-first order'
  - 'AC-007 zero-cost warm path'
slices:
  - slice_id: 'P2-S02-A'
    title: 'Red: failing tests for guard decision engine and orchestrator contract'
    status: '[PLANNED]'
    goal: red-testing
    estimate_hours: 1.5
    files_to_change:
      - 'scripts/agent-customization/cortex/cortex-health-guard.test.mjs'
      - 'scripts/agent-customization/cortex/cortex-self-heal.test.mjs'
    acceptance_criteria:
      - 'Suites fail with import/contract errors before implementation exists'
      - 'Cover: trigger-once under cooldown, backoff doubling, max-attempts T3 guidance data, lock create/heartbeat/release/reclaim, atomic JSON write-then-rename, dry-run command list, warm-path zero fs/spawn instrumentation'
    parallelizable: false
    dependencies: []
  - slice_id: 'P2-S02-B'
    title: 'Implement cortex-health-guard.mjs (probe adapter, state file, cooldown/backoff, lock mgmt)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2.5
    files_to_change:
      - 'scripts/agent-customization/cortex/cortex-health-guard.mjs'
    acceptance_criteria:
      - 'Exported API: evaluateSelfHeal({probe, now, stateDir, spawner}) returning {action, guidanceFields, spawnDecision} per the Architecture section'
      - 'State file schema exactly as specified; env knobs honored; kill switch respected'
    parallelizable: false
    dependencies: ['P2-S02-A']
  - slice_id: 'P2-S02-C'
    title: 'Implement cortex-self-heal.mjs orchestrator CLI + gitignore for state/lock files'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2.5
    files_to_change:
      - 'scripts/agent-customization/cortex/cortex-self-heal.mjs'
      - '.gitignore'
    acceptance_criteria:
      - '--dry-run --json reports composed commands touching only existing automation, classifier-first ordering'
      - 'Lock held whole run with heartbeat; outcome + measured duration appended to state history'
      - 'rag-index/data/cortex-self-heal-state.json, rag-index/data/cortex-self-heal.repair.lock, rag-index/data/*.lock ignored'
    parallelizable: false
    dependencies: ['P2-S02-B']
  - slice_id: 'P2-S02-D'
    title: 'Green: run suites, dry-run orchestrator, record evidence'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/rag-self-heal-hooks.plans.md'
    acceptance_criteria:
      - 'Both suites green under npm run jest:mjs'
      - 'Dry-run JSON output pasted into Latest validation evidence'
      - 'slice-advancement gate green for P2-S02 with the four changed files'
    parallelizable: false
    dependencies: ['P2-S02-C']
```

## Phase 3 — Server-integrated detection and model-facing guidance
**Status:** [PLANNED]

```yaml
phase: 3
title: 'Server-integrated detection and model-facing guidance'
status: '[PLANNED]'
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
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P3-S02 --changed-files=scripts/mcp-semantic/tools/search-corpus.mjs,scripts/mcp-semantic/tools/search-context.mjs,scripts/mcp-semantic/repo-cortex-mcp.mjs'
acceptance_criteria:
  - 'AC-001 response augmentation and trigger-once wiring'
  - 'AC-005 post-repair warm observation without server restart'
placeholder_steps:
  - step: 1
    title: 'Phase kickoff'
    goal: planning
    status: '[PLANNED]'
  - step: 2
    title: 'Search-tool and index_stats wiring (red-green, 4 slices)'
    goal: implementing
    status: '[PLANNED]'
```

### Phase 3, Step 02 — Search-tool and index_stats wiring (TDD)
**Status:** [PLANNED]

```yaml
phase: 3
step: 2
title: 'Search-tool and index_stats wiring (TDD)'
status: '[PLANNED]'
goal: implementing
expansion: 'slices'
auto_expand: true
tdd_sequence: 'red-green'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Phase 4, Step 01 — phase kickoff'
skills:
  - 'red-testing'
  - 'implementation-standards'
  - 'testing'
  - 'green-testing'
  - 'repo-cortex-embeddings'
specialists:
  - 'unit-test-writer'
  - 'implementation-executor'
constitution_check: 'pass — surgical wiring into existing tools; response shapes remain backward-compatible (additive self_heal field)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P3-S02-A'
validation:
  - 'npm run jest:mjs -- --testPathPatterns=search-corpus --runInBand --no-coverage'
  - 'npm run jest:mjs -- --testPathPatterns=search-context --runInBand --no-coverage'
  - 'npm run jest:mjs -- --testPathPatterns=repo-cortex-mcp --runInBand --no-coverage'
acceptance_criteria:
  - 'Degraded responses render T1/T2/T3 templates exactly; never throw into the search path on guard failure (fail-open to today''s behavior plus plain degraded fields)'
  - 'self_heal block present in both structured content and text summary'
slices:
  - slice_id: 'P3-S02-A'
    title: 'Red: failing tests for response augmentation and memo invalidation'
    status: '[PLANNED]'
    goal: red-testing
    estimate_hours: 1.5
    files_to_change:
      - 'scripts/mcp-semantic/tools/search-corpus.test.mjs'
      - 'scripts/mcp-semantic/tools/search-context.test.mjs'
      - 'scripts/mcp-semantic/repo-cortex-mcp.test.mjs'
    acceptance_criteria:
      - 'DENSE_FORCE_STATE=model-only simulations assert self_heal block + template text'
      - 'Memo-invalidation test: probe sequence model-only -> warm observed without process restart'
      - 'Guard-failure fail-open test: exception inside guard still returns plain BM25 results'
    parallelizable: false
    dependencies: ['P2-S02-D']
  - slice_id: 'P3-S02-B'
    title: 'Implement search-corpus.mjs wiring (guard call + cachedDenseReadiness invalidation)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'scripts/mcp-semantic/tools/search-corpus.mjs'
    acceptance_criteria:
      - 'Guard consulted only on degraded outcomes; memo honored while warm'
      - 'Degraded re-probe throttled to once per cooldown window'
    parallelizable: false
    dependencies: ['P3-S02-A']
  - slice_id: 'P3-S02-C'
    title: 'Implement search-context.mjs and index_stats wiring'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'scripts/mcp-semantic/tools/search-context.mjs'
      - 'scripts/mcp-semantic/repo-cortex-mcp.mjs'
    acceptance_criteria:
      - 'index_stats reports embeddings-vs-chunks mismatch fields consumed by the guard'
      - 'search-context mirrors search-corpus guidance behavior'
    parallelizable: false
    dependencies: ['P3-S02-B']
  - slice_id: 'P3-S02-D'
    title: 'Green: suites green, live degraded-run smoke evidence, plan updated'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/rag-self-heal-hooks.plans.md'
    acceptance_criteria:
      - 'Three suites green; one live DENSE_FORCE_STATE smoke transcript in Latest validation evidence'
      - 'slice-advancement gate green for P3-S02'
    parallelizable: false
    dependencies: ['P3-S02-C']
```

## Phase 4 — Server-unavailable guidance seam (facade)
**Status:** [PLANNED]

```yaml
phase: 4
title: 'Server-unavailable guidance seam (facade)'
status: '[PLANNED]'
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
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P4-S02 --changed-files=scripts/agent-customization/mcp/lazy-facade-core.mjs'
acceptance_criteria:
  - 'AC-006 T4 guidance on spawn failure'
placeholder_steps:
  - step: 1
    title: 'Phase kickoff'
    goal: planning
    status: '[PLANNED]'
  - step: 2
    title: 'Facade failure guidance (red-green, 3 slices)'
    goal: implementing
    status: '[PLANNED]'
```

### Phase 4, Step 02 — Facade failure guidance (TDD)
**Status:** [PLANNED]

```yaml
phase: 4
step: 2
title: 'Facade failure guidance (TDD)'
status: '[PLANNED]'
goal: implementing
expansion: 'slices'
auto_expand: true
tdd_sequence: 'red-green'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Phase 5, Step 01 — documentation'
skills:
  - 'red-testing'
  - 'implementation-standards'
  - 'green-testing'
specialists:
  - 'unit-test-writer'
  - 'implementation-executor'
constitution_check: 'pass — error-path enrichment only; no behavior change on the success path'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P4-S02-A'
validation:
  - 'npm run jest:mjs -- --testPathPatterns=lazy-facade --runInBand --no-coverage'
acceptance_criteria:
  - 'Spawn failure renders T4 template with manual recovery commands; error still propagates'
slices:
  - slice_id: 'P4-S02-A'
    title: 'Red: failing test for T4 facade guidance'
    status: '[PLANNED]'
    goal: red-testing
    estimate_hours: 1
    files_to_change:
      - 'scripts/agent-customization/mcp/lazy-facade-core.test.mjs'
    acceptance_criteria:
      - 'Injected spawn failure asserts T4 text, self_heal action unavailable, error propagation, and no retry loop'
    parallelizable: false
    dependencies: ['P3-S02-D']
  - slice_id: 'P4-S02-B'
    title: 'Implement T4 guidance in lazy-facade-core.mjs'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 1.5
    files_to_change:
      - 'scripts/agent-customization/mcp/lazy-facade-core.mjs'
    acceptance_criteria:
      - 'fallbackHint path replaced by template-rendered guidance while preserving JSON-RPC error semantics'
    parallelizable: false
    dependencies: ['P4-S02-A']
  - slice_id: 'P4-S02-C'
    title: 'Green: suite green + evidence recorded'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 0.5
    files_to_change:
      - 'plans/rag-self-heal-hooks.plans.md'
    acceptance_criteria:
      - 'slice-advancement gate green for P4-S02'
    parallelizable: false
    dependencies: ['P4-S02-B']
```

## Phase 5 — Documentation and closure
**Status:** [PLANNED]

```yaml
phase: 5
title: 'Documentation and closure'
status: '[PLANNED]'
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

### Phase 5, Step 01 — Documentation alignment
**Status:** [PLANNED]

```yaml
phase: 5
step: 1
title: 'Documentation alignment'
status: '[PLANNED]'
goal: documenting
expansion: 'none'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Phase 5, Step 02 — compression and closure'
skills:
  - '6-documentation-style'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P5-S01 --changed-files=.github/skills/repo-cortex-workflow/SKILL.md,.github/skills/repo-cortex-embeddings/SKILL.md'
acceptance_criteria:
  - 'repo-cortex-workflow gains a short Self-heal surface section (pointer only; canonical repair sequences stay single-sourced there)'
  - 'repo-cortex-embeddings degraded-state section links to the self_heal response contract'
  - 'No duplication of repair sequences into other docs'
files_to_change:
  - '.github/skills/repo-cortex-workflow/SKILL.md'
  - '.github/skills/repo-cortex-embeddings/SKILL.md'
```

### Phase 5, Step 02 — Compression and closure
**Status:** [PLANNED]

```yaml
phase: 5
step: 2
title: 'Compression and closure'
status: '[PLANNED]'
goal: logging
expansion: 'none'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'plan archive'
skills:
  - 'logging'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P5-S02 --changed-files=plans/rag-self-heal-hooks.plans.md,plans/README.md,plans/Roadmap.md'
acceptance_criteria:
  - 'PlanUpdate block appended; README/Roadmap flipped to [DONE] only when every phase is DONE; tracker moves to plans/completed per index policy'
```

## Latest validation evidence

```text
Authoring pass (2026-07-14, 01-planning authoring instance):
- Plan authored from live research (hooks contract, facade/server sources, gates, skills); files read listed in the authoring session log.
- Mechanism recommendation: Option B (server-internal core) + Option C shared state file; Option A demoted pending Step 02 empirical verification.
- Registration: pending at authoring time — README.md and Roadmap.md edits land in the same pass before the gate self-check.
Gate self-check (to be recorded after run):
- pending: node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P1-S01 --changed-files=plans/rag-self-heal-hooks.plans.md,plans/README.md,plans/Roadmap.md
Verification (independent instance, Phase 1 Step 03):
- status: pending — NO green-light recorded yet. Do not activate Phase 2+ until a verification instance records green-light: true here.
```

## Handoff query

```text
Resume the RAG self-heal plan: read plans/rag-self-heal-hooks.plans.md. Active position = Phase 1. If the research artifact plans/rag-self-heal-hooks.research.md is missing, dispatch Phase 1 Step 02 (researching). Otherwise run Phase 1 Step 03 verification: a fresh 01-planning verification instance must validate packets and record green-light: true (or blockers) in ## Latest validation evidence before Phase 2 activates.
```
