# Semantic Index — scripts/semantic-index/

A SQLite-backed retrieval system for the NeatapticTS repository. Two cooperating
layers provide complementary coverage:

- **BM25 layer** — full-text search over every textual surface: generated folder
  READMEs, skill files, agent files, plan trackers, demo source summaries, and key
  root docs. Exact keyword and phrase matching.
- **Dense embedding layer** — locally cached `all-MiniLM-L6-v2` ONNX
  sentence-transformer producing 384-dimensional float32 vectors for each corpus
  chunk. Enables semantic (paraphrase-tolerant) retrieval and hybrid ranking. The
  CLI enables this with `--dense`; the MCP `search_corpus` tool defaults
  `use_dense` to `true` and falls back honestly to BM25 when the dense runtime is
  not warm.

AI agents and MCP tools use this index to retrieve accurate, up-to-date context
without exhaustive codebase traversal. This directory is the **offline build-time
layer only**. MCP tool exposure is handled in
[Semantic_Knowledge_MCP_Tools.plans.md](../../plans/Semantic_Knowledge_MCP_Tools.plans.md).

---

## Artifacts

### BM25 index

| File | Role |
|---|---|
| `build-index.mjs` | CLI — scans corpus, chunks documents, writes `data/semantic-index.sqlite` |
| `query-index.mjs` | CLI — BM25 full-text query with family filter and JSON output |
| `validate-index.mjs` | CLI — asserts min row counts, freshness proofs, and max staleness |
| `init-schema.mjs` | Schema initializer — creates tables and FTS5 virtual table on first run |
| `schema.sql` | Source-of-truth schema definition (read by `init-schema.mjs`) |
| `chunker.mjs` | Text windowing — overlapping fixed-size windows + heading extraction |
| `ts-chunker.mjs` | CLI — ts-morph AST-level symbol chunker for `ts-source` family |
| `freshness.mjs` | Freshness prover — `(mtime_ms, file_size_bytes, sha256_hex)` triple |
| `cli-utils.mjs` | Shared CLI helpers — `parseCliArgs`, `printHelp`, `writeJsonOrText`, `fail` |
| `data/semantic-index.sqlite` | **Generated artifact** — gitignored, produced by `build-index.mjs` |

### Dense embedding layer

| File | Role |
|---|---|
| `download-model.mjs` | CLI — downloads ONNX model + tokenizer assets to `scripts/semantic-index/models/` |
| `embed-index.mjs` | CLI — runs ONNX inference, stores BLOB vectors in `data/embeddings.sqlite` |
| `prewarm-dense.mjs` | CLI — idempotent bootstrap that downloads the model when needed, embeds changed chunks, then validates the dense store |
| `dense-readiness.mjs` | CLI probe — reports `cold`, `model-only`, or `warm` with counts and a human-readable reason |
| `validate-embeddings.mjs` | CLI gate — asserts vector count matches corpus chunk count |
| `eval-embeddings.mjs` | CLI — MRR@5 evaluation of BM25-only vs hybrid retrieval quality |
| `hybrid-rank.mjs` | Library — BM25 + cosine linear combination with configurable alpha |
| `query-dense.mjs` | CLI — BM25 query with optional hybrid dense reranking |
| `code-quality-scanner.mjs` | CLI gate — flags missing/weak JSDoc and high-complexity symbols |
| `eval-queries.json` | Canonical 20-query eval set with hit criteria and MRR@5 threshold |
| `data/embeddings.sqlite` | **Generated artifact** — gitignored, BLOB vector store |

The generated database lives at `data/semantic-index.sqlite` (repo root) and is
excluded from version control via `.gitignore`. Rebuild it at any time with
`npm run index:build`. Prepare the dense sidecar with `npm run index:prewarm`
after the corpus exists.

---

## Bootstrap contract

Fresh clones do not contain `data/embeddings.sqlite` or
`scripts/semantic-index/models/` because both are generated, gitignored runtime
artifacts. After `npm install` and `npm run build`, run the dense bootstrap before
expecting MCP dense search to be warm:

```sh
npm run index:prewarm
```

The bootstrap is idempotent. It downloads the ONNX model only when
`scripts/semantic-index/models/model.onnx` is missing, incrementally embeds only
chunks whose content hash changed, then validates that the embedding count matches
the corpus chunk count.

Use the readiness probe to check the current state without issuing a search:

```sh
npm run index:dense-readiness
```

Example readiness outputs:

```json
{ "ready": false, "state": "cold", "reason": "Dense model assets are absent.", "chunk_count": null, "embedding_count": null }
```

```json
{ "ready": false, "state": "model-only", "reason": "Dense model is present but the embeddings database is absent.", "chunk_count": null, "embedding_count": null }
```

```json
{ "ready": true, "state": "warm", "reason": "All 29300 chunks have embeddings.", "chunk_count": 29300, "embedding_count": 29300 }
```

Readiness states map directly to MCP behavior:

| State | Meaning | MCP `search_corpus` behavior |
|---|---|---|
| `cold` | Model assets are absent. | Returns BM25 results with `dense_degraded: true`, `dense_state: "cold"`, and `dense_reason`. |
| `model-only` | Model exists, but embeddings are absent or incomplete. | Returns BM25 results with `dense_degraded: true`, `dense_state: "model-only"`, and `dense_reason`. |
| `warm` | Model exists and embedding count equals corpus chunk count. | Uses hybrid dense ranking by default and emits `dense_state: "warm"`. |

Rewarm whenever the corpus changes: source JSDoc regenerated into `src/**/README.md`,
plans, skills, agents, examples, benchmarks, or root docs. The standard rewarm
cycle is:

```sh
npm run index:build
npm run index:prewarm
npm run index:dense-readiness
```

For CI or orchestrator gates, use the standard readiness gate:

```sh
node scripts/agent-customization/gates/dense-readiness.gate.mjs --json
```

The gate passes only when readiness is `warm`; otherwise it returns a structured
failure with a fix hint that points operators back to `npm run index:prewarm`.

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

| Priority | Family label | Glob | Chunker |
|---|---|---|---|
| 1 | `readme` | `src/**/README.md` | Markdown heading-window |
| 2 | `ts-source` | `src/**/*.ts` (non-test, non-declaration) | ts-morph symbol-level (`ts-chunker.mjs`) |
| 3 | `skill` | `.github/skills/**/SKILL.md` | Markdown heading-window |
| 4 | `agent` | `.github/agents/*.agent.md` | Markdown heading-window |
| 5 | `plan` | `plans/**/*.md` (excluding `plans/completed/`) | Markdown heading-window |
| 6 | `completed-plan` | `plans/completed/**/*.md` | Markdown heading-window |
| 7 | `demo` | `examples/**/README.md`, `examples/**/*.ts` | Markdown heading-window |
| 8 | `benchmark` | `benchmarks/README.md`, `benchmarks/**/*.test.ts` | Markdown heading-window |
| 9 | `root-doc` | `README.md`, `CLAUDE.md`, `STYLEGUIDE.md`, `CONTRIBUTING.md` | Markdown heading-window |
| 10 | `copilot-instructions` | `.github/copilot-instructions.md` | Markdown heading-window |

The `ts-source` family uses the **ts-morph AST chunker** (`ts-chunker.mjs`) instead of
the text-window chunker. Each chunk covers one exported symbol (function, class,
interface, or type alias) with its full JSDoc comment, signature, and source path.
This enables semantic search to resolve queries like "NEAT mutation add node" directly
to the `mutate` function definition — vocabulary that does not appear verbatim in any
README.

Use `--family <label>` in `query-index.mjs` or `query-dense.mjs` to restrict results
to one family.

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
### ts-chunker.mjs

Extract `ts-source` family chunks from `src/**/*.ts` using ts-morph AST traversal.
Each exported symbol (function, class, interface, type alias) becomes one chunk
containing its JSDoc comment, signature, and source path.

```
node scripts/semantic-index/ts-chunker.mjs [options]

Options:
  --json            Emit JSON chunk output
  --source <path>   Limit scanning to one or more explicit source file paths
  --help            Show this help
```

**npm alias:** `npm run index:ts-chunk`

### download-model.mjs

Download the `all-MiniLM-L6-v2` ONNX model and tokenizer assets from Hugging Face
(`Xenova/all-MiniLM-L6-v2`) to `scripts/semantic-index/models/`. Verifies the
SHA-256 of `model.onnx` and writes `model-meta.json`. Run once before `embed-index.mjs`.

```
node scripts/semantic-index/download-model.mjs [options]

Options:
  --json                  Emit JSON summary { modelId, modelSha256, dimension, assets }
  --model-directory <p>   Override local model cache directory
  --model-id <id>         Override local model identifier
  --repository-id <id>    Override Hugging Face repository id
  --expected-sha256 <h>   Override expected SHA-256 for model.onnx
  --dimension <n>         Override embedding dimension written to model-meta.json
  --help                  Show this help
```

**npm alias:** `npm run index:download-model`

### embed-index.mjs

Build or incrementally update the dense embedding index. Reads corpus chunks from
`data/semantic-index.sqlite`, runs each chunk through the ONNX model (mean-pool →
L2-normalize → 384-dim float32 BLOB), and stores vectors in `data/embeddings.sqlite`.
Skips chunks whose `chunk_sha256` and `model_id` are unchanged (incremental rule).

```
node scripts/semantic-index/embed-index.mjs [options]

Options:
  --dry-run                  Count work without writing embeddings
  --json                     Emit JSON summary { embedded, skipped, queued, dryRun }
  --database <path>          Override corpus database path
  --embeddings-database <p>  Override embeddings database path
  --model-directory <path>   Override local model cache directory
  --model-id <id>            Override model identifier
  --dimension <n>            Override embedding dimension
  --model-sha256 <hex>       Override model SHA-256 value
  --help                     Show this help
```

**npm alias:** `npm run index:embed`

### validate-embeddings.mjs

Assert that the embedding index is complete: vector count for the active model must
equal the corpus chunk count. Emits the standard gate JSON contract.

```
node scripts/semantic-index/validate-embeddings.mjs [options]

Options:
  --json                     Emit the standard gate JSON contract
  --database <path>          Override corpus database path
  --embeddings-database <p>  Override embeddings database path
  --model-id <id>            Restrict validation to one model id
  --help                     Show this help
```

**npm alias:** `npm run index:validate-embeddings`

### prewarm-dense.mjs

Bootstrap dense search for a local checkout. The command ensures model assets exist,
embeds any missing or changed chunks, and validates the embedding count.

```
node scripts/semantic-index/prewarm-dense.mjs [options]

Options:
  --dry-run  Log the planned bootstrap steps without spawning subprocesses
  --json     Emit a machine-readable success or failure summary
  --help     Show help
```

**npm alias:** `npm run index:prewarm`

### dense-readiness.mjs

Report whether dense search can run without degrading to BM25-only.

```
node scripts/semantic-index/dense-readiness.mjs [options]

Options:
  --json                       Emit the readiness report as JSON
  --database <path>            Override the semantic-index corpus database path
  --embeddings-database <p>    Override the embeddings database path
  --model-directory <path>     Override the local dense model directory
  --model-id <id>              Override the embedding model identifier
  --help                       Show help
```

**npm alias:** `npm run index:dense-readiness`

### eval-embeddings.mjs

Evaluate hybrid retrieval quality using the canonical 20-query eval set. Reports
MRR@5 for BM25-only and hybrid modes. Exits non-zero when the hybrid improvement
is below the minimum threshold (+0.02 by default).

```
node scripts/semantic-index/eval-embeddings.mjs [options]

Options:
  --json                        Emit JSON evaluation output
  --alpha <n>                   Hybrid BM25 weight 0–1 (default: 0.5)
  --database <path>             Override corpus database path
  --embeddings-database <p>     Override embeddings database path
  --model-directory <path>      Override local model cache directory
  --model-id <id>               Override model identifier
  --query-file <path>           Override eval query set path
  --min-hybrid-improvement <n>  Override required hybrid MRR@5 gain (default: 0.02)
  --help                        Show this help
```

**npm alias:** `npm run index:eval`

### query-dense.mjs

Full-text BM25 query with optional hybrid dense reranking.

```
node scripts/semantic-index/query-dense.mjs --query "query text" [options]

Options:
  --query <text>              Query text (required)
  --dense                     Enable hybrid dense reranking
  --alpha <n>                 Hybrid BM25 weight 0–1 (default: 0.5)
  --limit <n>                 Maximum result count (default: 10)
  --family <name>             Restrict to one document family
  --database <path>           Override corpus database path
  --embeddings-database <p>   Override embeddings database path
  --model-directory <path>    Override local model cache directory
  --model-id <id>             Override model identifier
  --json                      Emit JSON results
  --help                      Show this help
```

**npm alias:** `npm run index:query-dense -- --query "text" --dense`

### code-quality-scanner.mjs

Scan `src/**/*.ts` (non-test) for quality signals: missing JSDoc, weak JSDoc
(fewer than 10 non-whitespace words), and high cyclomatic complexity. Emits the
standard gate JSON contract.

```
node scripts/semantic-index/code-quality-scanner.mjs [options]

Options:
  --json                       Emit the standard gate JSON contract
  --complexity-threshold <n>   Maximum allowed cyclomatic complexity (default: 10)
  --min-jsdoc-words <n>        Minimum non-whitespace JSDoc words (default: 10)
  --source <path>              Limit scanning to one or more source file paths
  --help                       Show this help
```

**npm alias:** `npm run index:code-quality`

### docs-quality.metrics.mjs

Run the canonical docs-quality metrics pipeline and write deterministic run artifacts.

```
node scripts/semantic-index/docs-quality/docs-quality.metrics.mjs [options]

Options:
  --json                       Emit full run payload as JSON
  --scope <src|paths>          Scope type (default: src)
  --source <path>              Explicit source path when --scope=paths (repeatable)
  --min-jsdoc-words <n>        Minimum non-whitespace JSDoc words (default: 10)
  --complexity-threshold <n>   Maximum allowed cyclomatic complexity (default: 10)
  --run-id <value>             Run folder ID under artifacts/docs-quality/runs/ (default: default)
  --help                       Show help
```

**npm alias:** `npm run docs:quality:metrics`

### docs-quality.compare.mjs

Compare two docs-quality runs with strict compatibility checks. Comparisons are
accepted only when all required dimensions match exactly.

```
node scripts/semantic-index/docs-quality/docs-quality.compare.mjs --left=<manifest> --right=<manifest> [--json]

Options:
  --left <manifestPath>        Left run manifest path (required)
  --right <manifestPath>       Right run manifest path (required)
  --json                       Emit comparison payload as JSON
  --help                       Show help
```

**npm alias:** `npm run docs:quality:compare`

Strict compare requirements and rejection reason codes:

| Requirement | Reason code on mismatch |
|---|---|
| `metricVersion` must match | `METRIC_VERSION_MISMATCH` |
| `threshold.minJsdocWords` and `threshold.complexityThreshold` must match | `THRESHOLD_MISMATCH` |
| `scopeType` must match (`src` vs `paths`) | `SCOPE_TYPE_MISMATCH` |
| `scopeDigest` must match | `SCOPE_DIGEST_MISMATCH` |
| `scannerVersion` must match | `SCANNER_VERSION_MISMATCH` |

### docs-quality-metrics.gate.mjs

Run CI-safe docs-quality contract checks (schema validity, deterministic ordering,
comparator guard behavior, CLI/MCP parity, and invalid-contract rejection).

```
node scripts/agent-customization/gates/docs-quality-metrics.gate.mjs [--json]
```

**npm alias:** `npm run docs:quality:gate`

### Docs-quality artifact layout

Each run is written under `artifacts/docs-quality/runs/<run-id>/` with three
canonical files:

| File | Purpose |
|---|---|
| `summary.json` | Aggregated counts (`evidenceCount`, issue totals) |
| `evidence.json` | Canonical normalized evidence rows plus evidence digest |
| `manifest.json` | Contract metadata and pointers to summary/evidence artifacts |

### Migration note: retire ad hoc measurement flows

Use the canonical `docs:quality:metrics` + `docs:quality:compare` +
`docs:quality:gate` commands for all current docs-quality checks.

The older ad hoc flow pattern (including `scripts/semantic-index/tmp-weak-scan.mjs`
and one-off scanner JSON parsing commands) is deprecated for routine measurement
and should only be referenced for historical migration context.

### Parallel-run baseline runbook

When running docs-quality in parallel batches, use one locked baseline and keep
all candidate runs comparable to that same baseline configuration.

1. Create and lock a baseline run id with final compare dimensions:
   `docs:quality:metrics -- --scope=... --min-jsdoc-words=... --complexity-threshold=... --run-id=<baseline-id>`
2. Run candidate batches in parallel from that baseline using unique run ids,
   without changing scope or thresholds.
3. Compare each candidate to the same baseline manifest via `docs:quality:compare`.
4. After reconciliation, run one final candidate with identical configuration to
   the locked baseline and compare once more before promotion.

---

## Dense embedding layer

### ONNX model and tokenizer assets

The embedding layer uses [`all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2)
from the [Xenova / Transformers.js](https://github.com/xenova/transformers.js) family.
The model is downloaded once to the local cache at `scripts/semantic-index/models/` and
is never accessed at query time after that.

**Required assets** (downloaded by `download-model.mjs`):

| File | Role |
|---|---|
| `model.onnx` | ONNX sentence-transformer graph — the inference model |
| `tokenizer.json` | WordPiece vocabulary and tokenization rules |
| `tokenizer_config.json` | Tokenizer configuration (max length, padding, etc.) |
| `special_tokens_map.json` | CLS, SEP, PAD, MASK token identifiers |
| `model-meta.json` | Sidecar written by `download-model.mjs` — `{ model_id, model_sha256, dimension }` |

The SHA-256 of `model.onnx` is verified on every download and stored in `model-meta.json`
as a reproducibility anchor. The `models/` directory is gitignored; every developer or
CI job that needs dense search must run `download-model.mjs` once.

### `chunk_embeddings` table schema

Stored in a separate `data/embeddings.sqlite` (gitignored), keeping vector blobs
out of the BM25 database. In-process cosine similarity is computed over float32 BLOBs
without any native SQLite vector extension (`sqlite-vec` is a conditional upgrade path
that requires Windows/CI compatibility confirmation).

| Column | Type | Notes |
|---|---|---|
| `chunk_id` | INTEGER | Foreign key → `data/semantic-index.sqlite` `chunks.chunk_id` |
| `embedding` | BLOB | Raw IEEE 754 float32 vector; dimension fixed per `model_id` |
| `chunk_sha256` | TEXT | Content hash of the chunk at embedding time (freshness / incremental key) |
| `model_id` | TEXT | Canonical model identifier, e.g. `all-MiniLM-L6-v2` |
| `model_sha256` | TEXT | SHA-256 of the ONNX model file (reproducibility anchor) |
| `dimension` | INTEGER | Embedding vector dimension (384 for MiniLM-L6-v2) |
| `embedded_at` | TEXT | ISO 8601 timestamp — used for staleness detection |

### Incremental embedding rule

`embed-index.mjs` **skips** a chunk when both conditions hold:

- `chunk_sha256` matches the stored value (chunk body has not changed since last embed).
- `model_id` matches the current model.

It **re-embeds** when either the chunk body changes (detected via SHA-256) or the model
is upgraded to a new version. This makes incremental rebuilds fast: only added or edited
source chunks require new ONNX inference.

### Hybrid rank formula

When `--dense` / `use_dense: true` is enabled, BM25 and cosine similarity scores are
linearly combined using:

$$\text{score}(d, q) = \alpha \cdot \text{BM25\_norm}(d, q) + (1 - \alpha) \cdot \cos(\mathbf{e}_d, \mathbf{e}_q)$$

where:
- $\text{BM25\_norm}$ is min-max normalized over the candidate set.
- $\mathbf{e}_d$ and $\mathbf{e}_q$ are L2-normalized float32 embedding vectors.
- $\alpha$ is the BM25 weight (default 0.5, configurable via `--alpha`).
- The candidate set is the union of the BM25 pool and the nearest-neighbour dense pool
  (so dense search can surface chunks with no lexical overlap with the query).

### Default-on MCP and CLI opt-in policy

The MCP `search_corpus` tool defaults `use_dense` to `true`. On a warm machine,
that means hybrid dense ranking is used without an explicit tool argument. On a
cold or model-only machine, the MCP server degrades to BM25 and reports
`dense_degraded`, `dense_state`, and `dense_reason` so callers can tell the
difference between a true hybrid result and a fallback.

The standalone CLI remains explicit so local debugging can choose BM25-only or
hybrid behavior per command:

| Context | Dense behavior |
|---|---|
| `query-dense.mjs` CLI | Pass `--dense` to enable hybrid reranking |
| `search_corpus` MCP tool | Defaults to `"use_dense": true`; pass `"use_dense": false` for BM25-only |
| Direct API call | `{ useDense: true }` in options |

BM25-only behavior is identical whether or not the embeddings database exists. Dense
retrieval adds semantic recall, but it requires the local model cache and a completed
embedding build; run `npm run index:prewarm` to make default-on MCP dense search warm.

### Eval methodology and MRR@5 results

Retrieval quality is measured using **Mean Reciprocal Rank at 5** (MRR@5):

$$\text{MRR@5} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank of first hit at } k \leq 5}$$

A hit at rank $k$ is counted when the result belongs to an `expected_doc_families`
member AND (when non-null) the chunk heading or symbol name contains the expected
substring. Zero is scored for queries with no hit in the top 5.

The eval set (`eval-queries.json`) contains **20 canonical queries** covering:
- BM25-friendly exact-match queries (`readme`, `plan`, `skill` families).
- Semantic paraphrase queries that benefit from dense retrieval.
- 16 queries targeting the `ts-source` family (TypeScript symbol lookup).

**Measured MRR@5 results** (full corpus, `all-MiniLM-L6-v2`, alpha = 0.5):

| Ranker | MRR@5 |
|---|---|
| BM25-only | 0.1875 |
| Hybrid (α = 0.5) | 0.300 |
| Improvement | +0.1125 (required minimum: +0.02) |

The gate script `cortex-embeddings.gate.mjs` enforces the +0.02 improvement threshold
and fails the gate if the hybrid MRR@5 drops below BM25 + 0.02 on any re-evaluation.

---

## Query examples

```sh
# Basic BM25 search
node scripts/semantic-index/query-index.mjs "Network activation"

# Restrict to TypeScript source symbols (ts-source family)
node scripts/semantic-index/query-index.mjs "NEAT mutation" --family ts-source

# Restrict to generated src/ READMEs
node scripts/semantic-index/query-index.mjs "NEAT mutation" --family readme

# Restrict to agent files, JSON output, top 5 results
node scripts/semantic-index/query-index.mjs "orchestrator handoff" --family agent --limit 5 --json

# Restrict to active plan trackers
node scripts/semantic-index/query-index.mjs "checkpointing restore" --family plan

# Restrict to skill files
node scripts/semantic-index/query-index.mjs "coverage tranche" --family skill

# Hybrid dense search (requires npm run index:prewarm)
node scripts/semantic-index/query-dense.mjs --query "NEAT activation forward pass" --dense --json

# Hybrid search restricted to ts-source family, top 5
node scripts/semantic-index/query-dense.mjs --query "speciation compatibility" --dense --family ts-source --limit 5 --json
```

---

## Rebuild flow after `npm run docs`

The generated `src/**/README.md` files (family `readme`) are the highest-priority
corpus source. After any JSDoc change that triggers a docs refresh, the BM25 index
will be stale for those files until you rebuild. The embedding index will also be
stale for chunks whose text changed.

**Standard rebuild sequence:**

```sh
# 1. Regenerate all src/**/README.md from JSDoc
npm run docs

# 2. Incrementally re-index only changed files (freshness proof detects the update)
npm run index:build

# 3. Incrementally re-embed only changed chunks and validate the dense store
npm run index:prewarm

# 4. Confirm index health
npm run index:validate
npm run index:dense-readiness
```

The incremental BM25 rebuild in step 2 typically re-indexes only the handful of changed
README files and completes in well under a second. The incremental embedding rebuild in
step 3 re-embeds only chunks whose `chunk_sha256` changed. Use `--force` on
`build-index.mjs` only when you need to reprocess every document from scratch (e.g.,
after changing the chunking window size or schema).

---

## Generated databases and .gitignore

Both SQLite databases are developer-tool artifacts, not tracked assets.

| Database | Path | Build command |
|---|---|---|
| BM25 index | `data/semantic-index.sqlite` | `npm run index:build` |
| Embedding vectors | `data/embeddings.sqlite` | `npm run index:embed` |

Both paths are excluded from version control via `.gitignore`. Each developer or CI
job that needs these databases must build them locally:

```sh
# Standard full setup
npm run docs            # generate src/**/README.md from JSDoc
npm run index:build     # build BM25 index
npm run index:prewarm   # download model if absent, build embeddings, validate readiness
```

---

## Validation expectations

A healthy BM25 index produced from the full NeatapticTS corpus contains approximately
**1249 documents** and **29,315 chunks** across the ten corpus families (including
`ts-source`).

```sh
node scripts/semantic-index/validate-index.mjs --json --min-documents 1000 --min-chunks 25000
```

Expected shape when healthy:

```json
{
  "ok": true,
  "pass": true,
  "documents": 1249,
  "chunks": 29315,
  "failures": []
}
```

For the embedding index (after `embed-index.mjs` completes):

```sh
node scripts/semantic-index/validate-embeddings.mjs --json
```

Expected output:

```json
{ "pass": true, "evidence": [], "fixHint": null, "owner": "05-green-testing" }
```

Run the full gate to confirm model presence, embedding count, and MRR@5 threshold:

```sh
node scripts/agent-customization/gates/cortex-embeddings.gate.mjs --json
```

If the index is missing or stale, `pass` will be `false` and `evidence` will list
the specific failing checks with `fixHint` guidance.

Run the readiness gate when CI only needs to prove that default-on MCP dense search
is warm:

```sh
node scripts/agent-customization/gates/dense-readiness.gate.mjs --json
```
