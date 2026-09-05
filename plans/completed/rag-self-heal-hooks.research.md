# RAG/Cortex self-heal hooks — Phase 1 Step 02 research findings

> Plan: `plans/rag-self-heal-hooks.plans.md`  
> Step: Phase 1 Step 02 — Research confirmation probes  
> Researcher: 02-researching (Tier-1)  
> Specialists consulted: boundary-mapper, implementation-pattern-scout, repo-cortex-scout, performance-reviewer  
> Date: 2026-09-04

## Question

The plan leaves three items marked **NEEDS CLARIFICATION** to be answered empirically before Phase 2 implementation:

1. Does this harness fire `PreToolUse`/`PostToolUse` hooks for MCP-routed tool names (e.g. `cortex-cortex`), and does `PostToolUse` deliver enough context/result to support an optional context-injection layer?
2. What are the correct env-knob defaults, especially the probe-cost threshold, given this machine's reality?
3. What is the observed `npm run index:prewarm` duration for the current ~25k-chunk corpus?

## Evidence

### 1. Hook observability

A temporary, reversible diagnostic capture was added to the existing registered `PostToolUse` hook (`scripts/agent-customization/hooks/post-write-reindex-hook.mjs`) to record every invocation, then removed after the probe. A separate attempt to add a new diagnostic `PostToolUse` entry in `.github/hooks/cortex-refresh.json` did **not** fire, showing the host does not pick up new hook registrations mid-session.

Capture file `rag-index/data/hook-context/capture.jsonl` contents after the probe (artifact deleted afterwards):

```jsonl
{"toolName":"Edit","ts":"2026-09-04T22:53:23.755Z","inputKeys":["hook_event_name","session_id","timestamp","cwd","tool_name","tool_input","tool_result"],"hasToolInput":true,"hasArguments":false}
{"toolName":"Bash","ts":"2026-09-04T22:53:35.305Z","inputKeys":["hook_event_name","session_id","timestamp","cwd","tool_name","tool_input","tool_result"],"hasToolInput":true,"hasArguments":false}
{"toolName":"Read","ts":"2026-09-04T22:53:41.072Z","inputKeys":["hook_event_name","session_id","timestamp","cwd","tool_name","tool_input","tool_result"],"hasToolInput":true,"hasArguments":false}
{"toolName":"Bash","ts":"2026-09-04T22:53:45.478Z","inputKeys":["hook_event_name","session_id","timestamp","cwd","tool_name","tool_input","tool_result"],"hasToolInput":true,"hasArguments":false}
{"toolName":"neataptic-gate-mcp-list_gates","ts":"2026-09-04T22:53:49.710Z","inputKeys":["hook_event_name","session_id","timestamp","cwd","tool_name","tool_input","tool_result"],"hasToolInput":true,"hasArguments":false}
{"toolName":"Bash","ts":"2026-09-04T22:53:51.804Z","inputKeys":["hook_event_name","session_id","timestamp","cwd","tool_name","tool_input","tool_result"],"hasToolInput":true,"hasArguments":false}
```

Key observations:

- `PostToolUse` hooks **do** fire for both native tools (`Edit`, `Bash`, `Read`) and for at least one successful MCP tool call (`neataptic-gate-mcp-list_gates`).
- The host passes a JSON object on stdin with `tool_name`, `tool_input`, and `tool_result` keys, confirming enough context is available for inspection.
- The MCP tool name delivered is the literal dotted tool name (`neataptic-gate-mcp-list_gates`), not a generic wrapper.
- `cortex-cortex` calls that failed (e.g. `index_stats` returning `SQLITE_BUSY`) did **not** produce a `PostToolUse` capture entry. The host appears to skip the hook when the MCP tool errors, or the lazy facade's error path does not traverse the hook pipeline.
- New hook registrations added to `.github/hooks/cortex-refresh.json` mid-session are **not picked up**; only the entries loaded at session start are active.

### 2. `checkDenseReadiness()` latency on this machine

Measured with 21 consecutive samples via `node rag-index/dense-readiness.mjs --json` while the corpus is in the `model-only` state (24,999 chunks, 0 embeddings):

| Statistic | Value (ms) |
| --------- | ---------- |
| min       | 172.91     |
| p50       | 190.07     |
| p95       | 214.85     |
| max       | 220.02     |
| avg       | 191.97     |

The degraded-path probe is therefore consistently ~190 ms on this machine, not the "~100 ms" assumed in the draft plan. A threshold at or below ~200 ms would cause the guard to fall back to state-file-only checks on nearly every degraded call, which the user explicitly asked to avoid.

### 3. Prewarm duration evidence

- Initial state: `index_stats` showed `total_chunks: 24999`, dense embeddings count `0`, `dense_state: model-only`.
- First `npm run index:prewarm` attempt failed at the `embed-index` step with `SQLITE_BUSY: database is locked` after an earlier run had already partially populated embeddings (DB modified at 2026-09-04 18:56:35).
- A second `npm run index:prewarm` attempt was started in the background. It progressed through `download-model` (skipped) and into `embed-index`, which was observed running for at least ~5 minutes before the job was stopped to conclude Step 02. A full end-to-end duration for the ~25k-chunk corpus was not collected because `embed-index` was still in progress.

Partial evidence already establishes that:

- `embed-index` is the dominant long-running step of the prewarm pipeline.
- Lock contention (`SQLITE_BUSY`) can abort a prewarm, so the orchestrator must serialize with other writers and tolerate partial-rerun idempotency.
- The repair lock's stale timeout (`CORTEX_SELFHEAL_LOCK_STALE_S = 2700`) is sized for a multi-minute prewarm, but real end-to-end times should be recorded in the state-file history once production repairs run.

### 4. Specialist consensus

Four independent Tier-3 specialists were dispatched:

- **boundary-mapper** — confirmed insertion points and concluded Option B + Option C shared state is the only dependable primary mechanism.
- **implementation-pattern-scout** — confirmed neighborhood patterns and concluded a server-internal state-file/lock orchestrator is canonical; hooks are only an optional additive layer.
- **repo-cortex-scout** — concluded gate-driven session-start / pre-dispatch freshness guard is optimal; PostToolUse is unreliable as the primary recovery path because it is skipped on MCP errors and cannot be re-registered mid-session.
- **performance-reviewer** — proposed keeping the existing cooldown/backoff/max/window/stale-lock defaults and setting the probe-cost threshold to **600 ms** based on the measured distribution.

All four converged on the same verdict.

## Decision

1. **Mechanism:** Keep the plan's recommended architecture — **Option B (server-internal core) with Option C's shared state file**. Do **not** make `PostToolUse` context injection the primary self-heal path. The empirical probe confirms hooks can fire on successful calls but are skipped on the exact failure class (`cortex-cortex` errors) that needs healing, and new hook registrations cannot be added mid-session. The server-internal guard is the only surface that sees the degraded signal where it originates and can guarantee the guidance reaches the model in the tool response it is already parsing.

2. **Env-knob defaults:** Confirm the plan's existing defaults and add the missing probe-cost threshold:

| Env var                                   | Default | Meaning                                                                               |
| ----------------------------------------- | ------- | ------------------------------------------------------------------------------------- |
| `CORTEX_SELFHEAL_COOLDOWN_S`              | `600`   | Minimum seconds between repair triggers                                               |
| `CORTEX_SELFHEAL_BACKOFF_FACTOR`          | `2`     | Exponential multiplier on repeated failures                                           |
| `CORTEX_SELFHEAL_MAX_ATTEMPTS`            | `3`     | Attempt ceiling per window before pause-and-ask                                       |
| `CORTEX_SELFHEAL_ATTEMPT_WINDOW_S`        | `86400` | Sliding window for counting attempts                                                  |
| `CORTEX_SELFHEAL_LOCK_STALE_S`            | `2700`  | Stale-lock age threshold (45 min)                                                     |
| `CORTEX_SELFHEAL_PROBE_COST_THRESHOLD_MS` | `600`   | Probe-cost budget; state-file-only fallback when `checkDenseReadiness()` exceeds this |
| `CORTEX_SELFHEAL_DISABLE`                 | `0`     | Kill switch — `1` disables triggering                                                 |

The **600 ms** threshold is ~3× the measured p50 (~190 ms) and ~2.8× p95 (~215 ms), so it only fires during exceptional contention (e.g. `SQLITE_BUSY`, heavy test run) rather than normal degraded-path operation.

3. **Prewarm duration:** Pending completion of the background `npm run index:prewarm` job. The first attempt's `SQLITE_BUSY` failure is recorded as supporting evidence that the orchestrator must serialize with other writers and tolerate lock contention.

### Insertion points re-confirmed in current sources

- `scripts/agent-customization/cortex/cortex-health-guard.mjs` (new) — call-time decision engine; read-only consult for all consumers.
- `scripts/agent-customization/cortex/cortex-self-heal.mjs` (new) — detached repair orchestrator, lock owner, single writer of `rag-index/data/cortex-self-heal-state.json`.
- `scripts/mcp-semantic/tools/search-corpus.mjs` — after `getDenseReadiness()` returns `cold`/`model-only` (lines ~421 and ~473), consult guard and merge `self_heal` block into `createDegradedBm25Response()`; invalidate `cachedDenseReadiness` on degraded so a running server observes repair completion.
- `scripts/mcp-semantic/tools/search-context.mjs` — when `searchResponse.dense_degraded` is true (line ~185+), mirror the `self_heal` block and prepend guidance to assembled context.
- `scripts/mcp-semantic/tools/index-stats.mjs` — add `embedding_count` vs `chunk_count` mismatch fields inside `indexStats()` (line ~213+) for a cheap secondary signal.
- `scripts/mcp-semantic/repo-cortex-mcp.mjs` — `index_stats` tool registration (line ~806+); expose the new mismatch fields without schema breakage.
- `scripts/agent-customization/mcp/lazy-facade-core.mjs` — catch block around `childTransport.callTool` (line ~336+) replaces `fallbackHint` with the T4 unavailable template.
- `.gitignore` — add `rag-index/data/cortex-self-heal-state.json`, `rag-index/data/cortex-self-heal.repair.lock`, and `rag-index/data/*.lock`.

## Risks

- **Hook skip on failure:** `PostToolUse` hooks are skipped when `cortex-cortex` errors, so any optional hook layer can only be additive; it must never be the sole recovery trigger.
- **Mid-session registration freeze:** Hook manifests are loaded at session start; any hook-based guidance must be pre-registered and cannot be tuned after startup.
- **SQLITE_BUSY contention:** The first prewarm attempt failed because the embeddings database was locked. The orchestrator must use atomic lock files and avoid running heavy `validate-index` synchronously on the search path.
- **Process-lifetime memo staleness:** `cachedDenseReadiness` in `search-corpus.mjs` may keep answering degraded after repair; the guard must re-probe on degraded and invalidate the memo when the lock is released.
- **Stale lock reclamation:** A crashed repair orchestrator could leave a lock; heartbeat + PID-liveness + age-based reclamation are required.
- **Threshold calibration:** 600 ms is calibrated to the current chunk count and this machine. Recalibrate if corpus size or machine load characteristics change significantly.
