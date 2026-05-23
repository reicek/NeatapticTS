# NeatChat Local Retrieval and Memory

**Status:** [PLANNED]

> NeatChat's own internal local DB, retrieval engine, and memory layer for better conversation
> quality. This is **separate from the Repo Cortex** (Semantic_Knowledge_Foundation and related
> plans). It may reuse vector/search patterns from the Repo Cortex, but it owns its own
> local database, demo runtime contracts, and memory semantics.

## Purpose

The NeatChat demo needs its own internal retrieval and memory system so conversation quality
can improve across sessions and across exchanges within a session. The W3 episodic memory
bank implemented in `NEATchat_Followup.plans.md` (now archived [DONE]) established a token-overlap
retrieval pattern. This plan extends and hardens that foundation into a durable, queryable
local memory store with a retrieval engine that respects NeatChat's demo-local runtime
contracts.

**This is not the Repo Cortex.** The Repo Cortex indexes NeatapticTS library documentation,
skills, agents, and plans for developer tooling. This plan indexes NeatChat's own conversation
history, learned associations, and session context for conversational quality. The two systems
share a no-conflicts boundary.

**Reuse boundary:** this plan may reuse chunking patterns, SQLite schema shapes, or BM25
query patterns from the Repo Cortex if they are available, but must not depend on the Repo
Cortex being present. NeatChat's memory must be self-contained and runnable in browser
(IndexedDB) and Node (SQLite) without the `data/semantic-index.sqlite` corpus.

## Non-goals

- Replacing or merging with the Repo Cortex (those plans remain separate).
- Changes to `src/` library code.
- External API calls for embeddings or memory (local-only).
- Conversational capabilities beyond the NeatChat demo scope.
- Modifying the snapshot generator or MCP tools.
- CI-mandatory test gates for the browser IndexedDB path (unit tests use Node mocks).

## Dependencies

- `examples/neatChat/` — existing NeatChat demo surfaces.
- `examples/neatChat/core/` — session services, safety services, routing services.
- Archived `NEATchat_Followup.plans.md` [DONE] — W3 episodic memory baseline to extend.
- Archived `plans/completed/neatChat-live-safety-red.plans.md` [DONE] — closed live-safety baseline; this plan extends
  memory only and must not reopen `runNeatChatExchange` safety-gate ownership.
- `better-sqlite3` or IndexedDB (browser) for local storage.

## Scope

### Memory system design

NeatChat memory has two tiers:

| Tier | Storage | Scope | Retention |
|---|---|---|---|
| Session memory | In-process (Map / array) | Current session only | Cleared on session reset |
| Durable memory | SQLite (Node) / IndexedDB (browser) | Persists across sessions | Pruned by LRU cap |

### Retrieval engine

- **Short-term recall:** exact token-overlap retrieval (reuses W3 pattern from archived plan).
- **Mid-term recall:** BM25 over stored conversation turns and learned associations.
- **Optional dense recall:** cosine similarity over precomputed embedding blobs (opt-in;
  requires ONNX model, mirrors the Repo Cortex Layer 5 opt-in policy).

### Artifacts

| Artifact | Path | Notes |
|---|---|---|
| Memory type definitions | `examples/neatChat/memory/neatChat.memory.types.ts` | MemoryEntry, MemoryQuery, MemoryResult |
| Memory DB wrapper | `examples/neatChat/memory/neatChat.memory.db.ts` | SQLite (Node) / IndexedDB (browser) adapter |
| Retrieval engine | `examples/neatChat/memory/neatChat.retrieval.ts` | BM25 + token-overlap retrieval |
| Memory services | `examples/neatChat/memory/neatChat.memory.services.ts` | Store, retrieve, prune, export |
| Memory constants | `examples/neatChat/memory/neatChat.memory.constants.ts` | Cap sizes, TTL, BM25 params |
| Memory unit tests | `examples/neatChat/memory/neatChat.memory.services.test.ts` | AAA, single-expect-per-test |
| Retrieval unit tests | `examples/neatChat/memory/neatChat.retrieval.test.ts` | AAA, single-expect-per-test |
| Session integration hook | `examples/neatChat/core/neatChat.session.services.ts` | Extend to read/write memory layer |
| Browser memory adapter | `examples/neatChat/memory/neatChat.memory.idb.ts` | IndexedDB adapter for browser runtime |

### SQLite schema (Node runtime)

```sql
-- Memory entries: one row per stored conversation fact or exchange summary
CREATE TABLE IF NOT EXISTS memory_entries (
  entry_id    INTEGER PRIMARY KEY,
  session_id  TEXT NOT NULL,
  entry_type  TEXT NOT NULL,    -- 'exchange', 'association', 'correction'
  content     TEXT NOT NULL,    -- stored text or JSON payload
  tokens      TEXT,             -- space-separated token list for overlap retrieval
  score       REAL DEFAULT 1.0, -- reinforcement score (higher = more relevant)
  created_at  INTEGER NOT NULL, -- unix ms
  last_used   INTEGER NOT NULL  -- unix ms (LRU pruning key)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
  content,
  tokens,
  content='memory_entries',
  content_rowid='entry_id',
  tokenize='porter unicode61'
);
```

### IndexedDB schema (browser runtime)

Object store `neatchat_memory`:
- Key: `entry_id` (auto-increment)
- Indexes: `session_id`, `last_used`, `entry_type`
- Same fields as the SQLite schema above.

### LRU pruning policy

When the entry count exceeds `MAX_MEMORY_ENTRIES` (default: 1000), prune the oldest entries
by `last_used` ascending until the count returns to the cap. The cap is configurable via
`neatChat.memory.constants.ts`.

### Integration with session services

The memory layer plugs into `examples/neatChat/core/neatChat.session.services.ts` as an
optional sidecar. The session services call `storeExchangeMemory(sessionId, exchange)` after
a successful live exchange and `retrieveMemoryContext(sessionId, query)` before candidate
selection to enrich the routing context. Both calls are guarded: if the memory DB is
unavailable, session services continue without memory context.

## Implementation phases

### Phase 1 — Planning [PLANNED]

#### Step 01 — Author step packets (01-planning) [PLANNED]

```yaml
phase: 1
step: 1
agent: "01-planning"
agent_file: ".github/agents/01-planning.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/NeatChat_Local_Retrieval_Memory.plans.md"
skills: "tracker-handoff, plan-sync-validation, neatchat-systems"
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NeatChat_Local_Retrieval_Memory.plans.md
```

**Step objective:** Read `examples/neatChat/core/` to understand the current session service
boundary. Read the archived `NEATchat_Followup.plans.md` W3 episodic memory notes. Read
`plans/completed/neatChat-live-safety-red.plans.md` to confirm no conflicts with the closed live-safety baseline.
Confirm `better-sqlite3` is available for Node use inside `examples/`. Author Step 02–07 packets.

### Phase 2 — Research [PLANNED]

#### Step 02 — Recon existing NeatChat memory boundary and W3 pattern (02-researching) [PLANNED]

```yaml
phase: 2
step: 2
agent: "02-researching"
agent_file: ".github/agents/02-researching.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/NeatChat_Local_Retrieval_Memory.plans.md"
skills: "neatchat-systems, plan-alignment"
validation:
  - Read examples/neatChat/core/ for session service integration points
  - Read plans/completed/NEATchat_Followup.plans.md W3 section for episodic memory baseline
  - Confirm examples/neatChat/memory/ does not already exist
  - Identify browser entry point for IndexedDB integration
```

**Step objective:** Map the session service integration points, identify the W3 token-overlap
retrieval function to extend, confirm the correct browser entry point for IndexedDB hooks, and
hand off a compact brief to Step 04.

### Phase 3 — Red tests [PLANNED]

#### Step 03 — Red contracts for memory store, retrieval, and pruning (03-red-testing) [PLANNED]

```yaml
phase: 3
step: 3
agent: "03-red-testing"
agent_file: ".github/agents/03-red-testing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/NeatChat_Local_Retrieval_Memory.plans.md"
skills: "red-test-contracts, neatchat-systems"
validation:
  - npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/memory
```

**Step objective:** Write failing tests for:
- `neatChat.memory.services.ts`: `storeExchangeMemory` stores an entry and returns the entry ID.
- `neatChat.retrieval.ts`: `retrieveMemoryContext` returns entries ranked by token-overlap score.
- `neatChat.memory.services.ts`: when entry count exceeds `MAX_MEMORY_ENTRIES`, pruning removes oldest entries.
- `neatChat.retrieval.ts`: retrieval returns empty array gracefully when memory is empty.
Each `it()` has exactly one top-level `expect(...)`.

### Phase 4 — Implementation [PLANNED]

#### Step 04 — Implement memory DB, retrieval engine, and session integration (04-implementing) [PLANNED]

```yaml
phase: 4
step: 4
agent: "04-implementing"
agent_file: ".github/agents/04-implementing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/NeatChat_Local_Retrieval_Memory.plans.md"
skills: "neatchat-systems, agent-script-tooling"
validation:
  - npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/memory
  - npx tsc --noEmit -p tsconfig.json
```

**Step objective:** Implement all artifacts in the Scope section:
1. `neatChat.memory.types.ts` — MemoryEntry, MemoryQuery, MemoryResult interfaces.
2. `neatChat.memory.constants.ts` — MAX_MEMORY_ENTRIES (1000), TTL, BM25 default params.
3. `neatChat.memory.db.ts` — SQLite adapter for Node (via `better-sqlite3`).
4. `neatChat.memory.idb.ts` — IndexedDB adapter for browser (uses async IDB API).
5. `neatChat.retrieval.ts` — token-overlap + BM25 retrieval over stored entries.
6. `neatChat.memory.services.ts` — store, retrieve, prune, export; uses DB adapter interface.
7. Integration hook in `examples/neatChat/core/neatChat.session.services.ts` — call memory store
   after live exchange; call retrieve before candidate selection (both guarded by availability check).
8. Do not touch `runNeatChatExchange` safety gate — that closed baseline lives in `plans/completed/neatChat-live-safety-red.plans.md`.

### Phase 5 — Green validation [PLANNED]

#### Step 05 — Validate memory services and session integration (05-green-testing) [PLANNED]

```yaml
phase: 5
step: 5
agent: "05-green-testing"
agent_file: ".github/agents/05-green-testing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/NeatChat_Local_Retrieval_Memory.plans.md"
skills: "green-validation-gates, neatchat-systems"
validation:
  - npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/memory
  - npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/core/neatChat.session.services
  - npx tsc --noEmit -p tsconfig.json
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NeatChat_Local_Retrieval_Memory.plans.md
```

**Step objective:** All memory unit tests green. Existing session service tests still pass
(memory integration is guarded so it cannot break the existing safety-gate tests). TypeScript
clean with no new errors.

### Phase 6 — Docs [PLANNED]

#### Step 06 — Document memory layer and retrieval API (06-documenting) [PLANNED]

```yaml
phase: 6
step: 6
agent: "06-documenting"
agent_file: ".github/agents/06-documenting.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/NeatChat_Local_Retrieval_Memory.plans.md"
skills: "educational-docs, neatchat-systems"
```

**Step objective:** Write `examples/neatChat/memory/README.md` documenting:
- How the memory layer differs from the Repo Cortex corpus index.
- The two-tier memory model (session vs durable).
- The SQLite schema and IndexedDB store shape.
- The LRU pruning policy.
- The retrieval ranking formula.
- How to clear or export the memory DB for debugging.
Add JSDoc to all exported functions in the memory module.

### Phase 7 — Logging [PLANNED]

#### Step 07 — Session log and plan closure (07-logging) [PLANNED]

```yaml
phase: 7
step: 7
agent: "07-logging"
agent_file: ".github/agents/07-logging.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/NeatChat_Local_Retrieval_Memory.plans.md"
skills: "tracker-handoff, summarizing-session-log"
```

## Acceptance criteria and validation gates

| Gate | Command | Expected |
|---|---|---|
| Memory unit tests green | `npx jest --testPathPattern=examples/neatChat/memory` | All pass |
| Session service tests unbroken | `npx jest --testPathPattern=examples/neatChat/core/neatChat.session.services` | All pass |
| TypeScript clean | `npx tsc --noEmit -p tsconfig.json` | 0 new errors |
| Store + retrieve round-trip | Unit test | Entry stored → retrieved with correct tokens |
| Pruning at cap | Unit test | After 1001 entries, count returns to MAX_MEMORY_ENTRIES |
| Empty retrieval graceful | Unit test | Returns `[]` without throwing when memory is empty |
| Memory layer is guarded | Unit test | Session services work normally when DB adapter returns null |
| No Repo Cortex dependency | Manual check | `examples/neatChat/memory/` imports nothing from `scripts/semantic-index/` |

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Active plan: plans/NeatChat_Local_Retrieval_Memory.plans.md [PLANNED]

Context:
  - This plan is SEPARATE from the Repo Cortex (Semantic_Knowledge_Foundation etc.)
  - It extends NeatChat's own internal conversation memory, not the library corpus index.
  - The W3 episodic memory baseline from NEATchat_Followup.plans.md (archived [DONE]) is
    the starting point. Read plans/completed/NEATchat_Followup.plans.md W3 section first.
  - Do not touch the safety gate in neatChat.session.services.ts (owned by `plans/completed/neatChat-live-safety-red.plans.md` [DONE]).

Key artifacts:
  examples/neatChat/memory/neatChat.memory.types.ts      (types)
  examples/neatChat/memory/neatChat.memory.db.ts         (SQLite adapter)
  examples/neatChat/memory/neatChat.memory.idb.ts        (IndexedDB adapter)
  examples/neatChat/memory/neatChat.retrieval.ts         (BM25 + token-overlap retrieval)
  examples/neatChat/memory/neatChat.memory.services.ts   (store, retrieve, prune, export)
  examples/neatChat/memory/neatChat.memory.services.test.ts (unit tests)
  examples/neatChat/memory/neatChat.retrieval.test.ts    (unit tests)

Boundary contract:
  - Memory layer must not import from scripts/semantic-index/ (no Repo Cortex dependency).
  - Session service integration must be guarded (memory unavailability does not break exchanges).

Start with Step 01 (01-planning): read examples/neatChat/core/ and the archived W3 notes
before authoring remaining step packets.

Focused test command:
  npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/memory

Plan sync check:
  node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NeatChat_Local_Retrieval_Memory.plans.md
```
