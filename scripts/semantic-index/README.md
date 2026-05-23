# Semantic Index — scripts/semantic-index/

A SQLite-backed BM25 full-text search index covering every textual surface of the
NeatapticTS repository: generated folder READMEs, skill files, agent files, plan
trackers, demo source summaries, and key root docs.

AI agents and MCP tools use this index to retrieve accurate, up-to-date context
without exhaustive codebase traversal. This directory is the **offline build-time
layer only**. MCP tool exposure is handled in
[Semantic_Knowledge_MCP_Tools.plans.md](../../plans/Semantic_Knowledge_MCP_Tools.plans.md).

---

## Artifacts

| File | Role |
|---|---|
| `build-index.mjs` | CLI entry point — scans corpus, chunks documents, writes SQLite |
| `query-index.mjs` | CLI — BM25 full-text query with family filter and JSON output |
| `validate-index.mjs` | CLI — asserts min row counts, freshness proofs, and max staleness |
| `init-schema.mjs` | Schema initializer — creates tables and FTS5 virtual table on first run |
| `schema.sql` | Source-of-truth schema definition (read by `init-schema.mjs`) |
| `chunker.mjs` | Text windowing — overlapping fixed-size windows + heading extraction |
| `freshness.mjs` | Freshness prover — `(mtime_ms, file_size_bytes, sha256_hex)` triple |
| `cli-utils.mjs` | Shared CLI helpers — `parseCliArgs`, `printHelp`, `writeJsonOrText`, `fail` |
| `data/semantic-index.sqlite` | **Generated artifact** — gitignored, produced by `build-index.mjs` |

The generated database lives at `data/semantic-index.sqlite` (repo root) and is
excluded from version control via `.gitignore`. Rebuild it at any time with
`npm run index:build`.

---

## SQLite schema

```sql
-- One row per scanned file
CREATE TABLE documents (
  doc_id      INTEGER PRIMARY KEY,
  file_path   TEXT    NOT NULL UNIQUE,   -- repo-relative path, forward slashes
  doc_family  TEXT    NOT NULL,          -- corpus family label (see below)
  mtime_ms    INTEGER NOT NULL,          -- file modification time, Unix ms
  file_size   INTEGER NOT NULL,          -- file size in bytes
  sha256      TEXT    NOT NULL,          -- SHA-256 hex of file content
  indexed_at  INTEGER NOT NULL           -- time this row was written, Unix ms
);

-- One row per text window
CREATE TABLE chunks (
  chunk_id     INTEGER PRIMARY KEY,
  doc_id       INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  chunk_index  INTEGER NOT NULL,         -- 0-based window index within the document
  heading_path TEXT,                     -- e.g. "## Scope > ### Artifacts"
  body_text    TEXT    NOT NULL,         -- raw window text (~2 KB, 128-token overlap)
  char_start   INTEGER NOT NULL,         -- byte offset into original document
  char_end     INTEGER NOT NULL
);

-- FTS5 virtual table (BM25 ranking via built-in SQLite scorer)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  body_text,
  heading_path,
  content='chunks',
  content_rowid='chunk_id',
  tokenize='porter unicode61'
);
```

Three triggers (`chunks_ai`, `chunks_ad`, `chunks_au`) keep `chunks_fts` in sync
with the `chunks` table automatically. A `documents_family_idx` index accelerates
family-filtered queries.

---

## Corpus families

Documents are collected in the following priority order and tagged with a family label:

| Priority | Family label | Glob |
|---|---|---|
| 1 | `readme` | `src/**/README.md` |
| 2 | `skill` | `.github/skills/**/SKILL.md` |
| 3 | `agent` | `.github/agents/*.agent.md` |
| 4 | `plan` | `plans/**/*.md` (excluding `plans/completed/`) |
| 5 | `completed-plan` | `plans/completed/**/*.md` |
| 6 | `demo` | `examples/**/README.md`, `examples/**/*.ts` |
| 7 | `benchmark` | `benchmarks/README.md`, `benchmarks/**/*.test.ts` |
| 8 | `root-doc` | `README.md`, `CLAUDE.md`, `STYLEGUIDE.md`, `CONTRIBUTING.md` |
| 9 | `copilot-instructions` | `.github/copilot-instructions.md` |

Use `--family <label>` in `query-index.mjs` to restrict results to one family.

---

## Freshness proof model

Every document row stores a triple: `(mtime_ms, file_size, sha256)`. On each
incremental rebuild the scanner computes the current triple from disk and compares
it against the stored row:

- **Match** → document is skipped (no re-chunk, no FTS update).
- **Mismatch** → old chunks are deleted, the document is re-chunked, and new chunks
  are inserted into `chunks` (triggers keep `chunks_fts` in sync automatically).
- **`--force` flag** → skips the freshness comparison and re-indexes every document.

The validator (`validate-index.mjs`) re-computes the current on-disk triple for
every stored document and fails if any row's triple no longer matches. Use
`--max-age-ms` to enforce a maximum staleness window (default: 24 hours).

---

## CLI reference

### build-index.mjs

Scan the corpus, chunk documents, and populate `data/semantic-index.sqlite`.

```
node scripts/semantic-index/build-index.mjs [options]

Options:
  --dry-run          Scan corpus without writing SQLite rows
  --force            Re-index unchanged documents even if freshness proof matches
  --json             Emit JSON summary { scanned, indexed, skipped, chunks }
  --database <path>  Path to SQLite database file (default: data/semantic-index.sqlite)
  --help             Show help
```

**npm alias:** `npm run index:build`

### query-index.mjs

Full-text BM25 search against the index.

```
node scripts/semantic-index/query-index.mjs "query text" [options]

Options:
  --query <text>     Query text (alternative to positional argument)
  --limit <n>        Maximum result count (default: 10)
  --family <name>    Restrict results to one document family
  --json             Emit JSON results
  --database <path>  Path to SQLite database file (default: data/semantic-index.sqlite)
  --help             Show help
```

**npm alias:** `npm run index:query -- "query text"`

### validate-index.mjs

Assert index health: minimum row counts, freshness proofs, and staleness age.

```
node scripts/semantic-index/validate-index.mjs [options]

Options:
  --json                Emit JSON validation result { ok, pass, documents, chunks, failures }
  --min-documents <n>   Minimum expected document rows (default: 1)
  --min-chunks <n>      Minimum expected chunk rows (default: 1)
  --max-age-ms <ms>     Maximum row age in milliseconds (default: 86400000 / 24 h)
  --database <path>     Path to SQLite database file (default: data/semantic-index.sqlite)
  --help                Show help
```

**npm alias:** `npm run index:validate`

---

## Query examples

```sh
# Basic BM25 search
node scripts/semantic-index/query-index.mjs "Network activation"

# Restrict to generated src/ READMEs
node scripts/semantic-index/query-index.mjs "NEAT mutation" --family readme

# Restrict to agent files, JSON output, top 5 results
node scripts/semantic-index/query-index.mjs "orchestrator handoff" --family agent --limit 5 --json

# Restrict to active plan trackers
node scripts/semantic-index/query-index.mjs "checkpointing restore" --family plan

# Restrict to skill files
node scripts/semantic-index/query-index.mjs "coverage tranche" --family skill
```

---

## Rebuild flow after `npm run docs`

The generated `src/**/README.md` files (family `readme`) are the highest-priority
corpus source. After any JSDoc change that triggers a docs refresh, the index
will be stale for those files until you rebuild.

**Standard rebuild sequence:**

```sh
# 1. Regenerate all src/**/README.md from JSDoc
npm run docs

# 2. Incrementally re-index only changed files (freshness proof detects the update)
npm run index:build

# 3. Confirm index health
npm run index:validate
```

The incremental rebuild in step 2 typically re-indexes only the handful of changed
README files and completes in well under a second for a standard docs run. Use
`--force` only when you need to reprocess every document from scratch (e.g., after
changing the chunking window size or schema).

---

## Generated DB and .gitignore

`data/semantic-index.sqlite` is a developer-tool artifact, not a tracked asset.
It is excluded from version control via the `.gitignore` entry:

```
data/semantic-index.sqlite
```

Each developer or CI job that needs the index must build it locally with
`npm run index:build` after running `npm run docs` at least once so the generated
`src/**/README.md` files are present.

---

## Validation expectations

A healthy index produced from the full NeatapticTS corpus (as of 2026-05-23) contains
approximately **830 documents** and **27,700+ chunks** across the nine corpus families.
Use the following command as a quick sanity check:

```sh
node scripts/semantic-index/validate-index.mjs --json --min-documents 800 --min-chunks 20000
```

Expected output shape when healthy:

```json
{
  "ok": true,
  "pass": true,
  "documents": 830,
  "chunks": 27726,
  "failures": []
}
```

If the index is missing or stale, `pass` will be `false` and `failures` will list
the specific failing documents or constraint violations.
