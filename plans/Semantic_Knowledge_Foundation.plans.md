# Semantic Knowledge Foundation (Repo Cortex — Layer 1)

**Status:** [PLANNED]

> The foundational layer for a no-compromise semantic helping system for the NeatapticTS repo.
> All higher-level Repo Cortex plans depend on this layer. It must be implemented first.

## Purpose

Build a SQLite-backed semantic corpus index that covers every textual surface of the
NeatapticTS repository: generated folder READMEs, skill files, agent files, plan trackers,
demo source summaries, and key source modules. The index enables fast BM25 full-text search
with freshness-proof validation so AI agents and MCP tools can retrieve accurate, up-to-date
context rather than relying solely on broad codebase traversal.

This plan covers only the **offline build-time layer**: scanning, chunking, indexing, and
freshness validation. It does not expose MCP tools (see
[Semantic_Knowledge_MCP_Tools.plans.md](Semantic_Knowledge_MCP_Tools.plans.md)) or the
browser-facing snapshot (see
[Semantic_Knowledge_Browser_Snapshot.plans.md](Semantic_Knowledge_Browser_Snapshot.plans.md)).

## Non-goals

- MCP server or tool exposure (handled in `Semantic_Knowledge_MCP_Tools.plans.md`).
- Browser-facing snapshot or IndexedDB loader (handled in `Semantic_Knowledge_Browser_Snapshot.plans.md`).
- Dense embedding or vector search (handled in `Semantic_Knowledge_Embeddings.plans.md`).
- NeatChat-internal conversational memory (separate plan: `NeatChat_Local_Retrieval_Memory.plans.md`).
- Changes to `src/` library code.
- CI-mandatory step — the index is a developer-tool artifact, not a test gate.

## Dependencies

- Node.js `better-sqlite3` or equivalent SQLite driver (existing in dev deps or added here).
- `crypto` built-in for SHA-256 freshness proofs.
- Generated `src/**/README.md` files must be present (run `npm run docs` first).
- No changes to `src/`, `testing/`, or `benchmarks/`.

## Scope

### Corpus sources

The scanner collects chunks from the following document families, in priority order:

| Priority | Source family | Glob |
|---|---|---|
| 1 | Generated folder READMEs | `src/**/README.md` |
| 2 | Skill files | `.github/skills/**/SKILL.md` |
| 3 | Agent files | `.github/agents/*.agent.md` |
| 4 | Active plan trackers | `plans/**/*.md` (excluding `plans/completed/`) |
| 5 | Completed plan baselines | `plans/completed/**/*.md` |
| 6 | Demo source summaries | `examples/**/README.md`, top-level JSDoc blocks from `examples/**/*.ts` |
| 7 | Benchmark summaries | `benchmarks/README.md` if present, key test file headers |
| 8 | Root docs | `README.md`, `CLAUDE.md`, `STYLEGUIDE.md`, `CONTRIBUTING.md` |
| 9 | Copilot instructions | `.github/copilot-instructions.md` |

### Chunking strategy

- Split each document into overlapping fixed-size text windows (default: 512 tokens / ~2 KB, 128-token overlap).
- Preserve section headings in chunk metadata so queries can identify the originating section.
- Emit one row per chunk: `(doc_id, chunk_index, heading_path, body_text, char_start, char_end)`.

### Freshness proof

Each document entry stores `(mtime_ms, file_size_bytes, sha256_hex)`. On rebuild, the scanner
compares the current file triple against the stored triple; documents whose triple matches are
skipped (no re-chunk). Full rebuilds are supported via `--force` flag.

### BM25 full-text search

Use SQLite FTS5 with BM25 ranking (`MATCH` queries via `fts5` virtual table). No external
search engine is required.

### Artifacts

| Artifact | Path | Notes |
|---|---|---|
| Main scanner | `scripts/semantic-index/build-index.mjs` | Entry point; orchestrates all phases |
| SQLite schema | `scripts/semantic-index/schema.sql` | Source-of-truth schema definition |
| Schema initializer | `scripts/semantic-index/init-schema.mjs` | Runs schema.sql on first start |
| Chunker utility | `scripts/semantic-index/chunker.mjs` | Text windowing + heading extraction |
| Freshness prover | `scripts/semantic-index/freshness.mjs` | mtime + size + SHA-256 triple |
| BM25 query CLI | `scripts/semantic-index/query-index.mjs` | CLI for manual search + validation |
| Validation gate | `scripts/semantic-index/validate-index.mjs` | Assert index freshness + row counts |
| Generated SQLite DB | `data/semantic-index.sqlite` | Gitignored; built by scanner |
| `.gitignore` entry | `.gitignore` | Add `data/semantic-index.sqlite` |
| `package.json` scripts | `package.json` | `index:build`, `index:query`, `index:validate` |

### SQLite schema (abbreviated)

```sql
-- documents table: one row per scanned file
CREATE TABLE IF NOT EXISTS documents (
  doc_id      INTEGER PRIMARY KEY,
  file_path   TEXT NOT NULL UNIQUE,
  doc_family  TEXT NOT NULL,        -- 'readme', 'skill', 'agent', 'plan', 'demo', ...
  mtime_ms    INTEGER NOT NULL,
  file_size   INTEGER NOT NULL,
  sha256      TEXT NOT NULL,
  indexed_at  INTEGER NOT NULL      -- unix ms
);

-- chunks table: one row per text window
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id     INTEGER PRIMARY KEY,
  doc_id       INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  chunk_index  INTEGER NOT NULL,
  heading_path TEXT,                -- e.g. "## Scope > ### Artifacts"
  body_text    TEXT NOT NULL,
  char_start   INTEGER NOT NULL,
  char_end     INTEGER NOT NULL
);

-- FTS5 virtual table for BM25
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  body_text,
  heading_path,
  content='chunks',
  content_rowid='chunk_id',
  tokenize='porter unicode61'
);
```

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
source_of_truth: "plans/Semantic_Knowledge_Foundation.plans.md"
skills: "tracker-handoff, plan-sync-validation"
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Semantic_Knowledge_Foundation.plans.md
```

**Step objective:** Verify roadmap alignment, identify any existing SQLite or corpus-scanning
utilities in `scripts/`, confirm `better-sqlite3` availability, and update this plan with any
changes before dispatching Step 02 through Step 07.

### Phase 2 — Research [PLANNED]

#### Step 02 — Recon existing scripts and deps (02-researching) [PLANNED]

```yaml
phase: 2
step: 2
agent: "02-researching"
agent_file: ".github/agents/02-researching.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Foundation.plans.md"
skills: "plan-alignment"
validation:
  - Confirm better-sqlite3 or equivalent is available in package.json devDependencies
  - Confirm scripts/semantic-index/ does not already exist
  - Read scripts/ root to identify reusable patterns from existing MCP scripts
```

**Step objective:** Read `scripts/` root and `package.json` to identify reusable helpers,
confirm SQLite driver availability, check for any prior corpus-scanning attempts, and hand
off a compact reconnaissance brief to Step 04 (implementing).

### Phase 3 — Red tests [PLANNED]

#### Step 03 — Red contracts for scanner + freshness + query (03-red-testing) [PLANNED]

```yaml
phase: 3
step: 3
agent: "03-red-testing"
agent_file: ".github/agents/03-red-testing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Foundation.plans.md"
skills: "red-test-contracts, creating-unit-tests"
validation:
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index
```

**Step objective:** Write failing unit tests for:
- `freshness.mjs`: given a known file, produces the correct `(mtime_ms, size, sha256)` triple.
- `chunker.mjs`: given sample markdown, emits expected chunk count and heading_path values.
- `validate-index.mjs`: fails when the index is empty or stale, passes when fresh.
Prefer owner-local test files under `scripts/semantic-index/` using the project Jest config.

### Phase 4 — Implementation [PLANNED]

#### Step 04 — Build scanner, schema, chunker, freshness, query CLI (04-implementing) [PLANNED]

```yaml
phase: 4
step: 4
agent: "04-implementing"
agent_file: ".github/agents/04-implementing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Foundation.plans.md"
skills: "plan-alignment, agent-script-tooling"
validation:
  - node scripts/semantic-index/build-index.mjs --dry-run
  - node scripts/semantic-index/query-index.mjs "Network activation"
  - node scripts/semantic-index/validate-index.mjs --json
```

**Step objective:** Implement all artifacts listed in the Scope section:
1. `schema.sql` and `init-schema.mjs`.
2. `freshness.mjs` — mtime + size + SHA-256 triple comparison.
3. `chunker.mjs` — overlapping window + heading extraction.
4. `build-index.mjs` — orchestrates scan, chunk, FTS5 population, and freshness skip.
5. `query-index.mjs` — CLI: `--query`, `--limit`, `--family`, `--json` output.
6. `validate-index.mjs` — asserts min row counts and max staleness age.
7. Add `data/semantic-index.sqlite` to `.gitignore`.
8. Add `index:build`, `index:query`, `index:validate` npm scripts to `package.json`.
All scripts must be non-interactive, support `--help`, and emit JSON when `--json` is passed.

### Phase 5 — Green validation [PLANNED]

#### Step 05 — Validate scanner, BM25 queries, freshness proofs (05-green-testing) [PLANNED]

```yaml
phase: 5
step: 5
agent: "05-green-testing"
agent_file: ".github/agents/05-green-testing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Foundation.plans.md"
skills: "green-validation-gates, running-unit-tests"
validation:
  - node scripts/semantic-index/build-index.mjs
  - node scripts/semantic-index/query-index.mjs "NEAT activation" --json
  - node scripts/semantic-index/validate-index.mjs --json
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index
```

**Step objective:** Run the full build pipeline end-to-end against the real repo corpus.
Confirm that:
- The index builds without errors.
- `query-index.mjs "NEAT activation"` returns ranked results from at least three document families.
- `validate-index.mjs` passes with fresh index.
- A second build run with no file changes skips all documents (freshness skip works).
- All unit tests from Step 03 are green.

### Phase 6 — Docs [PLANNED]

#### Step 06 — Document scripts with --help and README (06-documenting) [PLANNED]

```yaml
phase: 6
step: 6
agent: "06-documenting"
agent_file: ".github/agents/06-documenting.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Foundation.plans.md"
skills: "educational-docs"
validation:
  - node scripts/semantic-index/build-index.mjs --help
  - node scripts/semantic-index/query-index.mjs --help
  - node scripts/semantic-index/validate-index.mjs --help
```

**Step objective:** Add `--help` output to all CLI scripts. Write a short
`scripts/semantic-index/README.md` that describes the index schema, corpus families, freshness
model, and how to rebuild after a `npm run docs` refresh.

### Phase 7 — Logging [PLANNED]

#### Step 07 — Session log and plan closure handoff (07-logging) [PLANNED]

```yaml
phase: 7
step: 7
agent: "07-logging"
agent_file: ".github/agents/07-logging.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Foundation.plans.md"
skills: "tracker-handoff, summarizing-session-log"
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Semantic_Knowledge_Foundation.plans.md
```

**Step objective:** Record a compressed done-state entry, confirm that
`Semantic_Knowledge_MCP_Tools.plans.md` is ready to proceed, and update this tracker to
`[DONE]` or leave it `[WIP]` with a handoff query for the next session.

## Acceptance criteria and validation gates

| Gate | Command | Expected |
|---|---|---|
| Index builds | `node scripts/semantic-index/build-index.mjs` | Exit 0; rows inserted |
| BM25 search returns results | `node scripts/semantic-index/query-index.mjs "NEAT activation" --json` | JSON array with ≥ 3 results |
| Freshness skip works | Run build twice; compare row insert counts | Second run inserts 0 new rows |
| Validation gate passes | `node scripts/semantic-index/validate-index.mjs --json` | `{ pass: true }` |
| Unit tests green | `npx jest --config=jest.config.mjs --testPathPattern=scripts/semantic-index` | All pass |
| Help flags work | Each script `--help` | Exits 0 with usage text |
| Gitignore clean | `git status data/` | `data/semantic-index.sqlite` is ignored |

## Coverage backlog

- [PLANNED] Freshness prover unit tests.
- [PLANNED] Chunker unit tests with markdown fixtures.
- [PLANNED] Validation gate unit tests (empty index → fail; fresh index → pass).
- [PLANNED] Integration smoke: build → query → validate full pipeline.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Active plan: plans/Semantic_Knowledge_Foundation.plans.md [PLANNED]

Goal: Implement the SQLite-backed semantic corpus index for NeatapticTS (Repo Cortex Layer 1).

The plan covers:
- scripts/semantic-index/build-index.mjs (scanner + FTS5 population)
- scripts/semantic-index/schema.sql (SQLite schema)
- scripts/semantic-index/chunker.mjs (overlapping window + heading extraction)
- scripts/semantic-index/freshness.mjs (mtime + size + SHA-256 freshness proofs)
- scripts/semantic-index/query-index.mjs (BM25 CLI)
- scripts/semantic-index/validate-index.mjs (validation gate)
- data/semantic-index.sqlite (gitignored generated artifact)

Start with Step 01 (01-planning): verify roadmap alignment and confirm better-sqlite3
availability in package.json before dispatching Step 02 through 07.

Validation command after build:
  node scripts/semantic-index/validate-index.mjs --json

Plan sync check:
  node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Semantic_Knowledge_Foundation.plans.md
```
