# NeatChat Local Retrieval and Memory

**Status:** [WIP]

> NeatChat's own internal local DB, retrieval engine, and memory layer for better conversation
> quality. This is **separate from the Repo Cortex** (Semantic_Knowledge_Foundation and related
> plans). It may reuse vector/search patterns from the Repo Cortex, but it owns its own
> local database, demo runtime contracts, and memory semantics.

## Purpose

The NeatChat demo needs its own internal retrieval and memory system so conversation quality
can improve across sessions and across exchanges within a session. The W3 episodic memory
workstream (archived in `NEATchat_Followup.plans.md` [DONE]) established a token-overlap
retrieval pattern backed by an in-process episodic bank. This plan extends that foundation
into a **durable, queryable local memory store** with a two-tier retrieval engine, while
explicitly treating the W3 baseline files as already delivered and frozen.

**This is not the Repo Cortex.** The Repo Cortex indexes NeatapticTS library documentation,
skills, agents, and plans for developer tooling. This plan indexes NeatChat's own conversation
history, learned associations, and session context for conversational quality. The two systems
share a no-conflicts boundary.

**Reuse boundary:** this plan may reuse chunking patterns, SQLite schema shapes, or BM25
query patterns from the Repo Cortex, but must not depend on the Repo Cortex being present.
NeatChat memory must be self-contained and runnable in browser (IndexedDB) and Node (SQLite)
without `data/semantic-index.sqlite`.

## W3 Baseline Boundary (already delivered — do not re-implement)

The following files were delivered by the archived W3 workstream of `NEATchat_Followup.plans.md`
and are the baseline from which this plan extends:

| File                               | Location                  | Role                                                                                                                 |
| ---------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `neatChat.memory.types.ts`         | `examples/neatChat/core/` | `NeatChatMemoryRecord`, `NeatChatEpisodicMemoryBank`, query/result types                                             |
| `neatChat.memory.services.ts`      | `examples/neatChat/core/` | `createNeatChatEpisodicMemoryBank`, `addNeatChatMemoryRecord`, `retrieveNeatChatMemories`, `pruneNeatChatMemoryBank` |
| `neatChat.memory.services.test.ts` | `examples/neatChat/core/` | Existing W3 test suite — must remain green                                                                           |

These files use **in-process storage only** (Map/array). This plan adds durable storage as a
separate subsystem under `examples/neatChat/memory/`. Step 02 must verify that no new exported
symbol in `examples/neatChat/memory/` collides with the existing exports from
`examples/neatChat/core/neatChat.memory.services.ts` or `neatChat.memory.types.ts`.

## Non-goals

- Replacing or merging the W3 in-process baseline (those files remain in `core/`).
- Replacing or merging with the Repo Cortex (those plans remain separate).
- Changes to `src/` library code.
- External API calls for embeddings or memory (local-only).
- Conversational capabilities beyond the NeatChat demo scope.
- Modifying the snapshot generator or MCP tools.
- CI-mandatory test gates for the browser IndexedDB path (unit tests use the injected Node adapter).
- Reopening `runNeatChatExchange` safety ownership (closed in `plans/completed/neatChat-live-safety-red.plans.md`).
- Adding `fake-indexeddb` as a test dependency.

## Dependencies

- `examples/neatChat/` — existing NeatChat demo surfaces.
- `examples/neatChat/core/` — session services, safety services, routing services, and W3 baseline.
- Archived `NEATchat_Followup.plans.md` [DONE] — W3 episodic memory baseline (already delivered).
- Archived `plans/completed/neatChat-live-safety-red.plans.md` [DONE] — closed live-safety baseline;
  this plan extends memory only and must not reopen `runNeatChatExchange` safety-gate ownership.
- `better-sqlite3` — Node.js synchronous SQLite driver for durable memory (Node path).
- **Browser path libraries (TBD — Step 02 decision):** Dexie.js and MiniSearch are external design
  references only; neither is currently in `package.json`. Step 02 decides whether to add one or
  both as dependencies, or to use raw IndexedDB API with a local 50-line BM25 scorer instead.

## Design Anchors

External references that inform the durable memory design:

| Decision                                  | Rationale                                                                                                                                                                                                                                                            | Source                                                                                            |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **BM25 retrieval**                        | Best-Match 25 is the standard probabilistic ranking function used in SQLite FTS5. Proven effective for local full-text retrieval without neural overhead.                                                                                                            | [Wikipedia — Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)                                |
| **SQLite FTS5 in Node**                   | `better-sqlite3` exposes a synchronous SQLite API; FTS5 virtual tables provide built-in BM25 ranking via the `bm25()` auxiliary function.                                                                                                                            | [better-sqlite3 docs](https://github.com/WiseLibs/better-sqlite3)                                 |
| **Browser storage: Dexie.js (candidate)** | Dexie.js is a minimal Promise-based IndexedDB wrapper with typed schema management and range queries. A design reference and candidate — Step 02 decides whether to add it as a dependency based on bundle and runtime tradeoffs; raw IndexedDB API is the fallback. | [Dexie.js docs](https://dexie.org/)                                                               |
| **Browser BM25: zero-dependency scorer**  | MiniSearch provides in-browser BM25 with zero dependencies and fast indexed search. A design reference and candidate for the browser path — Step 02 decides between MiniSearch and a 50-line local BM25 scorer based on bundle tradeoffs.                            | [MiniSearch (npm)](https://www.npmjs.com/package/minisearch)                                      |
| **FTS5 content-table triggers**           | Without triggers, FTS5 content tables go stale on UPDATE or DELETE. Standard pattern: `after_insert`, `after_update`, `after_delete` triggers on `memory_entries` keep `memory_fts` synchronized.                                                                    | [SQLite FTS5 — external content tables](https://www.sqlite.org/fts5.html#external_content_tables) |
| **FTS query sanitization**                | Raw user input passed to FTS5 can cause syntax errors. The Repo Cortex Layer 1 plan shipped `sanitizeFtsQuery` in `scripts/semantic-index/`; the NeatChat memory layer must provide its own local copy and must not import from `scripts/semantic-index/`.           | Internal convention                                                                               |
| **Default DB path**                       | `data/neatchat-memory.sqlite` — distinct from `data/semantic-index.sqlite` so the two corpora cannot interfere. Path must be configurable for tests.                                                                                                                 | Internal convention                                                                               |
| **Adapter injection**                     | The memory services module accepts a `MemoryAdapter` interface so tests can inject an in-memory Node adapter without touching the filesystem or adding `fake-indexeddb`.                                                                                             | Internal convention                                                                               |

## Scope

### Memory system design

NeatChat memory has two tiers:

| Tier           | Storage                             | Scope                    | Retention                |
| -------------- | ----------------------------------- | ------------------------ | ------------------------ |
| Session memory | In-process (W3 baseline in `core/`) | Current session only     | Cleared on session reset |
| Durable memory | SQLite (Node) / IndexedDB (browser) | Persists across sessions | Pruned by LRU cap        |

The W3 in-process tier is **already delivered**. This plan adds the durable tier only.

### Retrieval engine

- **Short-term recall:** token-overlap retrieval (delegates to W3 baseline in `core/`).
- **Mid-term recall (Node):** BM25 via SQLite FTS5 `bm25()` over stored conversation turns.
- **Mid-term recall (browser):** in-memory BM25 scoring over the capped IndexedDB corpus loaded at session start (scorer choice — MiniSearch or a local 50-line implementation — decided in Step 02).
- **Optional dense recall:** cosine similarity over precomputed embedding blobs (opt-in; requires ONNX model, mirrors Repo Cortex Layer 5). Out of scope for the initial implementation.

### BM25 and FTS5 strategy

**Node path:** SQLite FTS5 with `tokenize='porter unicode61'`. The `bm25()` auxiliary function
provides ranking. FTS5 is used as a content table (`content='memory_entries'`), kept in
sync with the base table via three triggers (see schema below).

**Browser path:** At session start, load the capped corpus from IndexedDB into memory.
Run BM25 scoring using a zero-dependency local scorer. Step 02 decides between MiniSearch
(external candidate, not yet in `package.json`) and a 50-line local BM25 implementation over
pre-tokenized arrays, based on bundle size and runtime tradeoffs. The BM25 requirement is
non-negotiable; only the scorer implementation is open.

**FTS query sanitization:** The memory module must include its own `sanitizeFtsQuery(raw: string): string`
function. This function strips or escapes FTS5-special characters so user-controlled input
cannot inject FTS5 syntax errors. It must not import from `scripts/semantic-index/`.

### Artifacts

| Artifact                    | Path                                                        | Notes                                                                         |
| --------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Memory type definitions     | `examples/neatChat/memory/neatChat.memory.types.ts`         | `MemoryEntry`, `MemoryQuery`, `MemoryResult`, `MemoryAdapter` interface       |
| Memory DB wrapper (Node)    | `examples/neatChat/memory/neatChat.memory.db.ts`            | `better-sqlite3` SQLite adapter with FTS5 schema and triggers                 |
| Memory DB wrapper (browser) | `examples/neatChat/memory/neatChat.memory.idb.ts`           | IndexedDB adapter — Dexie.js or raw IDB, decided in Step 02                   |
| FTS query sanitizer         | `examples/neatChat/memory/neatChat.memory.fts.ts`           | Local `sanitizeFtsQuery`; no import from `scripts/semantic-index/`            |
| Retrieval engine            | `examples/neatChat/memory/neatChat.retrieval.ts`            | BM25 + token-overlap; Node uses FTS5, browser uses in-memory scorer           |
| Memory services             | `examples/neatChat/memory/neatChat.memory.services.ts`      | `storeExchangeMemory`, `retrieveMemoryContext`, `pruneMemory`, `exportMemory` |
| Memory constants            | `examples/neatChat/memory/neatChat.memory.constants.ts`     | `MAX_DURABLE_ENTRIES` (1000), TTL, BM25 params, `DEFAULT_DB_PATH`             |
| Memory unit tests           | `examples/neatChat/memory/neatChat.memory.services.test.ts` | AAA, single-expect-per-test; uses injected in-memory adapter                  |
| Retrieval unit tests        | `examples/neatChat/memory/neatChat.retrieval.test.ts`       | AAA, single-expect-per-test                                                   |
| Session integration hook    | `examples/neatChat/core/neatChat.session.services.ts`       | Guarded sidecar calls; does not reopen safety ownership                       |

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

-- FTS5 content-table sync triggers (keeps memory_fts consistent on insert/update/delete)
CREATE TRIGGER IF NOT EXISTS memory_fts_ai AFTER INSERT ON memory_entries BEGIN
  INSERT INTO memory_fts(rowid, content, tokens) VALUES (new.entry_id, new.content, new.tokens);
END;

CREATE TRIGGER IF NOT EXISTS memory_fts_ad AFTER DELETE ON memory_entries BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, content, tokens)
    VALUES ('delete', old.entry_id, old.content, old.tokens);
END;

CREATE TRIGGER IF NOT EXISTS memory_fts_au AFTER UPDATE ON memory_entries BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, content, tokens)
    VALUES ('delete', old.entry_id, old.content, old.tokens);
  INSERT INTO memory_fts(rowid, content, tokens) VALUES (new.entry_id, new.content, new.tokens);
END;
```

### IndexedDB schema (browser runtime)

Object store `neatchat_memory` (managed via Dexie.js or raw IndexedDB API — library choice decided in Step 02):

- Key: `entry_id` (auto-increment)
- Indexes: `session_id`, `last_used`, `entry_type`
- Same fields as the SQLite schema above.

### Default DB path

`data/neatchat-memory.sqlite` — distinct from `data/semantic-index.sqlite`. Configurable via
`DEFAULT_DB_PATH` constant or via the adapter constructor argument. Tests must override the
path or inject a fully in-memory adapter.

### LRU pruning policy

When the entry count exceeds `MAX_DURABLE_ENTRIES` (default: 1000), prune the oldest entries
by `last_used` ascending until the count returns to the cap. The cap is configurable via
`neatChat.memory.constants.ts`.

### Adapter injection and testing strategy

The `MemoryAdapter` interface (in `neatChat.memory.types.ts`) defines the methods the memory
services module calls. Two real adapters implement it: `SqliteMemoryAdapter` and `IdbMemoryAdapter`.
Tests inject a lightweight Node-only in-memory adapter (plain Map + array) that satisfies the
interface without touching the filesystem. **Do not add `fake-indexeddb` as a test dependency.**
The browser adapter is exercised manually or in a future browser test harness.

### Integration with session services

The memory layer plugs into `examples/neatChat/core/neatChat.session.services.ts` as a
**guarded optional sidecar**. The session service:

1. Calls `storeExchangeMemory(sessionId, exchange)` after a successful live exchange.
2. Calls `retrieveMemoryContext(sessionId, query)` before candidate selection to enrich routing context.

Both calls are wrapped in a `try/catch` guard: if the memory DB is unavailable or throws,
the session service continues without memory context and without surfacing the error to the user.

**Safety ownership is not reopened.** The `runNeatChatExchange` safety gate remains exactly
as closed in `plans/completed/neatChat-live-safety-red.plans.md`. Memory integration is
strictly additive and sits outside the safety critical path.

## Implementation phases

### Phase 1 — Planning [WIP]

#### Step 01 — Author step packets (01-planning) [WIP]

```yaml
phase: 1
step: 1
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[WIP]'
mode: 'sequential'
source_of_truth: 'plans/NeatChat_Local_Retrieval_Memory.plans.md'
skills: 'tracker-handoff, plan-sync-validation, neatchat-systems'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NeatChat_Local_Retrieval_Memory.plans.md
```

**Step objective:** This plan rewrite is the Step 01 output. Step 02–07 packets are authored
below. The planning rewrite: confirmed W3 baseline files under `examples/neatChat/core/`,
added Design Anchors with external references, clarified FTS5 triggers and adapter injection,
set [WIP] status, and tightened all phase objectives.

Documentation evidence (2026-05-24): Completed a focused weak-JSDoc cleanup pass on three top
files under `src/architecture/network/` with behavior preserved. Validation used
`node scripts/semantic-index/code-quality-scanner.mjs --json` parsing. Baseline weak count
from prior session artifact: `115`. Post-pass scanner weak count: `52` (delta: `-63`).

Documentation evidence (2026-05-24, tranche follow-up): Hardened the next weak-JSDoc hotspots
in `src/architecture/network/activate/network.activate.helpers.utils.ts`,
`src/architecture/network/activate/network.activate.contexts.utils.ts`, and
`src/architecture/network/standalone/network.standalone.utils.graph.ts` using docs-only edits
with no behavior/signature changes. Validation command:
`node scripts/semantic-index/code-quality-scanner.mjs --json` (PowerShell-parsed). Weak count
before this tranche: `52`. Weak count after this tranche: `43` (delta: `-9`).

Documentation evidence (2026-05-24, tranche continuation): Updated weak-JSDoc hotspots in
`src/architecture/network/training/network.training.utils.types.ts`,
`src/architecture/network/prune/network.prune.utils.types.ts`, and
`src/architecture/network/training/network.training.smoothing.utils.ts` with docs-only
comment improvements and no runtime behavior changes. Validation command:
`node scripts/semantic-index/code-quality-scanner.mjs --json` (PowerShell-parsed). Weak count
before this tranche: `43`. Weak count after this tranche: `37` (delta: `-6`). File diagnostics
via `get_errors` returned no errors on all edited files.

### Phase 2 — Research [PLANNED]

#### Step 02 — Recon NeatChat session boundary and verify export-collision free (02-researching) [PLANNED]

```yaml
phase: 2
step: 2
agent: '02-researching'
agent_file: '.github/agents/02-researching.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/NeatChat_Local_Retrieval_Memory.plans.md'
skills: 'neatchat-systems, plan-alignment'
validation:
  - Read examples/neatChat/core/ exports and confirm no collision with planned examples/neatChat/memory/ symbols
  - Confirm better-sqlite3 is in package.json dependencies or devDependencies
  - Confirm examples/neatChat/memory/ does not yet exist
  - Read examples/neatChat/core/neatChat.session.services.ts for session integration points
  - Decide browser storage library (raw IndexedDB vs Dexie.js) and browser BM25 scorer
    (local 50-line implementation vs MiniSearch) based on bundle size and runtime tradeoffs;
    record the decision in this plan before Step 04 begins
```

**Step objective:** Map session service integration points (exact locations for guarded memory
sidecar calls). Enumerate all exported symbols from `examples/neatChat/core/neatChat.memory.services.ts`
and `neatChat.memory.types.ts` and confirm none collide with the planned `examples/neatChat/memory/`
export surface. Confirm `better-sqlite3` is available. **Decide the browser-path library choice:**
evaluate raw IndexedDB API plus a local BM25 scorer versus adding Dexie.js and/or MiniSearch as
new `package.json` dependencies, weighing bundle size and runtime tradeoffs; record the decision
in the plan so Step 04 implements the chosen path. Hand off a compact brief to Step 04.

### Phase 3 — Red tests [PLANNED]

#### Step 03 — Red contracts for durable memory store, retrieval, and pruning (03-red-testing) [PLANNED]

```yaml
phase: 3
step: 3
agent: '03-red-testing'
agent_file: '.github/agents/03-red-testing.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/NeatChat_Local_Retrieval_Memory.plans.md'
skills: 'red-test-contracts, neatchat-systems'
validation:
  - npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/memory
```

**Step objective:** Write failing tests using an injected in-memory adapter (no filesystem,
no `fake-indexeddb`) for:

- `storeExchangeMemory` stores an entry and returns a truthy entry ID.
- `retrieveMemoryContext` returns entries ranked by BM25/token-overlap score descending.
- Pruning: when entry count exceeds `MAX_DURABLE_ENTRIES`, oldest entries are removed first.
- Empty retrieval: returns `[]` without throwing when the memory store is empty.
- Guard: when the adapter's `store` method throws, the session-service wrapper does not propagate the error.
- `sanitizeFtsQuery`: strips or escapes FTS5-special characters (e.g. `"`, `(`, `)`, `*`, `^`) from raw user input.

Each `it()` has exactly one top-level `expect(...)`.

### Phase 4 — Implementation [PLANNED]

#### Step 04 — Implement durable memory DB, retrieval engine, and session integration (04-implementing) [PLANNED]

```yaml
phase: 4
step: 4
agent: '04-implementing'
agent_file: '.github/agents/04-implementing.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/NeatChat_Local_Retrieval_Memory.plans.md'
skills: 'neatchat-systems, agent-script-tooling'
validation:
  - npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/memory
  - npx tsc --noEmit -p tsconfig.json
```

**Step objective:** Implement all artifacts in the Scope section:

1. `neatChat.memory.types.ts` — `MemoryEntry`, `MemoryQuery`, `MemoryResult`, `MemoryAdapter` interface.
2. `neatChat.memory.constants.ts` — `MAX_DURABLE_ENTRIES` (1000), TTL, BM25 params, `DEFAULT_DB_PATH = 'data/neatchat-memory.sqlite'`.
3. `neatChat.memory.fts.ts` — local `sanitizeFtsQuery(raw: string): string`; no import from `scripts/semantic-index/`.
4. `neatChat.memory.db.ts` — `SqliteMemoryAdapter` for Node (via `better-sqlite3`), including the FTS5 schema and the three insert/update/delete sync triggers.
5. `neatChat.memory.idb.ts` — `IdbMemoryAdapter` for browser using the storage library chosen in Step 02 (Dexie.js or raw IndexedDB API).
6. `neatChat.retrieval.ts` — token-overlap + BM25 retrieval; Node path uses FTS5 `bm25()`, browser path scores over in-memory corpus.
7. `neatChat.memory.services.ts` — `storeExchangeMemory`, `retrieveMemoryContext`, `pruneMemory`, `exportMemory`; accepts `MemoryAdapter` as a constructor/factory argument.
8. Guarded integration in `examples/neatChat/core/neatChat.session.services.ts` — `try/catch` sidecar calls after live exchange and before candidate selection; must not touch `runNeatChatExchange` safety gate.

Do not introduce any exported symbol that collides with `examples/neatChat/core/neatChat.memory.services.ts` or `neatChat.memory.types.ts`.

### Phase 5 — Green validation [PLANNED]

#### Step 05 — Validate durable memory services and session integration (05-green-testing) [PLANNED]

```yaml
phase: 5
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-testing.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/NeatChat_Local_Retrieval_Memory.plans.md'
skills: 'green-validation-gates, neatchat-systems'
validation:
  - npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/memory
  - npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/core/neatChat.session.services
  - npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/core/neatChat.memory.services
  - npx tsc --noEmit -p tsconfig.json
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NeatChat_Local_Retrieval_Memory.plans.md
```

**Step objective:** All new durable memory unit tests green. W3 baseline tests in
`examples/neatChat/core/neatChat.memory.services.test.ts` still pass (unchanged). Existing
session service tests still pass (memory integration is guarded so it cannot break the
existing safety-gate tests). TypeScript clean with no new errors. Plan sync gate passes.

### Phase 6 — Docs [PLANNED]

#### Step 06 — Document durable memory layer and retrieval API (06-documenting) [PLANNED]

```yaml
phase: 6
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-documenting.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/NeatChat_Local_Retrieval_Memory.plans.md'
skills: 'educational-docs, neatchat-systems'
```

**Step objective:** Write `examples/neatChat/memory/README.md` documenting:

- How the durable memory layer differs from the W3 in-process baseline in `core/`.
- How it differs from the Repo Cortex corpus index.
- The two-tier memory model (session W3 baseline vs durable new module).
- The SQLite schema, FTS5 triggers, and IndexedDB store shape.
- The `MemoryAdapter` injection pattern for tests.
- The LRU pruning policy and `MAX_DURABLE_ENTRIES` cap.
- The BM25 retrieval ranking formula, Node vs browser path distinction, and FTS sanitization requirement.
- The default DB path and how to override it.
- How to clear or export the memory DB for debugging.

Add JSDoc to all exported functions in the memory module. Include a Mermaid data-flow diagram
showing session service → memory sidecar → SQLite/IDB storage paths.

### Phase 7 — Logging [PLANNED]

#### Step 07 — Session log and plan closure (07-logging) [PLANNED]

```yaml
phase: 7
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-logging.agent.md'
status: '[PLANNED]'
mode: 'sequential'
source_of_truth: 'plans/NeatChat_Local_Retrieval_Memory.plans.md'
skills: 'tracker-handoff, summarizing-session-log'
```

## Acceptance criteria and validation gates

| Gate                            | Command                                                                                                                       | Expected                                                                             |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Durable memory unit tests green | `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/memory`                         | All pass                                                                             |
| W3 baseline tests unbroken      | `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/core/neatChat.memory.services`  | All pass                                                                             |
| Session service tests unbroken  | `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/core/neatChat.session.services` | All pass                                                                             |
| TypeScript clean                | `npx tsc --noEmit -p tsconfig.json`                                                                                           | 0 new errors                                                                         |
| Store + retrieve round-trip     | Unit test                                                                                                                     | Entry stored → retrieved with correct ranking                                        |
| Pruning at cap                  | Unit test                                                                                                                     | After 1001 entries, count returns to `MAX_DURABLE_ENTRIES`                           |
| Empty retrieval graceful        | Unit test                                                                                                                     | Returns `[]` without throwing when memory is empty                                   |
| Guard holds on adapter error    | Unit test                                                                                                                     | Session services continue when adapter throws                                        |
| FTS sanitization                | Unit test                                                                                                                     | `sanitizeFtsQuery` strips FTS5-special chars from raw user input                     |
| No Repo Cortex dependency       | Manual / import check                                                                                                         | `examples/neatChat/memory/` imports nothing from `scripts/semantic-index/`           |
| No symbol collision             | Step 02 recon output                                                                                                          | No export from `memory/` collides with `core/neatChat.memory.*` exports              |
| Default DB path distinct        | Code review                                                                                                                   | `DEFAULT_DB_PATH` is `data/neatchat-memory.sqlite`, not `data/semantic-index.sqlite` |
| Plan sync gate                  | `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NeatChat_Local_Retrieval_Memory.plans.md`        | `{pass: true}`                                                                       |

## Session evidence (docs-only hardening)

- 2026-05-24: docs-only JSDoc hardening loop executed on `src/**` with `node scripts/semantic-index/code-quality-scanner.mjs --json` as the effective local equivalent of MCP `scan_code_quality`.
- Weak JSDoc count reduced from 37 to 0 across focused hotspot batches (3-5 files per batch until final residual closure).
- Edited files were validated with file diagnostics; no new errors were reported in modified files.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Active plan: plans/NeatChat_Local_Retrieval_Memory.plans.md [WIP]
Active step: Phase 1 Step 01 [WIP] — planning rewrite complete; proceed to Step 02.

Context:
  - This plan is SEPARATE from the Repo Cortex (Semantic_Knowledge_Foundation etc.)
  - It adds a DURABLE memory tier; the W3 in-process baseline already lives in:
      examples/neatChat/core/neatChat.memory.types.ts
      examples/neatChat/core/neatChat.memory.services.ts
      examples/neatChat/core/neatChat.memory.services.test.ts
    Do not re-implement those files.
  - New durable subsystem lives under examples/neatChat/memory/.
  - Step 02 must confirm no symbol collision between core/ and memory/ exports.
  - Do not touch the safety gate in neatChat.session.services.ts
    (owned by plans/completed/neatChat-live-safety-red.plans.md [DONE]).
  - DB path: data/neatchat-memory.sqlite (not data/semantic-index.sqlite).
  - BM25: Node uses SQLite FTS5 bm25(); browser uses in-memory scorer (MiniSearch is a candidate design reference, not yet in package.json — Step 02 decides MiniSearch vs local 50-line implementation).
  - FTS sanitization: local sanitizeFtsQuery in neatChat.memory.fts.ts; no import from scripts/semantic-index/.
  - FTS5 triggers: three content-table sync triggers (insert, update, delete) are required.
  - Adapter injection: MemoryAdapter interface; tests use in-memory adapter, no fake-indexeddb.

Focused test commands:
  npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/memory
  npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPattern=examples/neatChat/core/neatChat.memory.services
  npx jest --config=jest.config.mjs --no-cache --runInBand --runTestsByPath scripts/semantic-index/code-quality/code-quality-scanner.red.test.ts

Plan sync check:
  node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NeatChat_Local_Retrieval_Memory.plans.md
```
