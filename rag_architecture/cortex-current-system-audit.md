# Cortex Current System Audit (Layers 1–6)

> Extracted from `plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md` for permanent reference.

> **Backing database.** The Repo Cortex is backed by a single consolidated Turso (libSQL) database accessed via the fully async `@libsql/client` driver (default local embedded replica `data/turso-replica.sqlite`; cloud primary `libsql://<db>.turso.io`). Vectors use native Turso vectors with `F8_BLOB` 8-bit quantization, approximate nearest neighbor search runs server-side via DiskANN (`libsql_vector_idx`, `vector_top_k()`), and hybrid ranking is performed SQL-side via Reciprocal Rank Fusion (RRF, k=60). The historical design content below describes the pre-Turso architecture that was subsequently migrated to this stack.

This document contains the complete audit of the NeatapticTS Repo Cortex system (Layers 1–6) as of 2026-06-08, identifying gaps against advanced RAG requirements.

## Current system audit

### What exists (Layers 1–6) — Updated with measured baseline

| Capability       | Current state                                                                                                                                                                | Gap severity                                              |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Corpus storage   | Turso (libSQL) consolidated database with native vectors (`F8_BLOB` quantization), 1,408 docs, 31,396 chunks across 10 families (1,407 docs, 31,363 chunks as of last build) | **Baseline sufficient**                                   |
| Full-text search | BM25 via FTS5 (porter+unicode61 tokenizer), heading-based markdown chunks (2,048-char max + 512-char overlap), per-symbol TS chunks (NO max size)                            | **CRITICAL — naive chunking destroys semantic coherence** |
| Dense embeddings | `all-MiniLM-L6-v2` (384-dim, 512-token max), 31,106 embeddings in BLOB storage, mean-pooled + L2-normalized                                                                  | **Small model, truncation issue with oversized chunks**   |
| Hybrid ranking   | Fixed alpha=0.5 blend of min-max-normalized BM25 + cosine similarity; BM25 MRR@5=0.225, Hybrid MRR@5=0.308, improvement=+0.083                                               | **No query-adaptive weighting, no cross-encoder**         |
| Family filtering | Single `family` parameter on `search_corpus` with 10 families (readme, ts-source, skill, agent, plan, completed-plan, demo, benchmark, root-doc, copilot-instructions)       | **No structured metadata filtering**                      |
| Result loading   | `load_chunk` by ID, `load_document` by path — no multi-source context assembly                                                                                               | **No context assembly, no dedup, no budget**              |
| Freshness        | mtime + size + SHA-256 freshness proofs, incremental rebuild                                                                                                                 | **Baseline sufficient**                                   |
| MCP tools        | 7 tools (search_corpus, load_chunk, load_document, freshness_check, index_stats, list_families, scan_code_quality)                                                           | **Missing advanced RAG tools**                            |
| Code quality     | JSDoc + complexity scanning via `scan_code_quality`                                                                                                                          | **Narrow scope**                                          |
| Browser snapshot | `docs/assets/semantic-snapshot.json` + IndexedDB loader                                                                                                                      | **Read-only, no browser-side search**                     |

### What is missing (Layer 7+ requirements)

| Missing capability                | Why it matters                                                                                                                                                                                                                  | Architecture complexity                                                    |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Semantic chunking**             | Heading-based splitting breaks code mid-function (ts-source chunks reach 42K chars!), loses AST context, produces 51.4% of chunks with no heading_path, and 2,604 ts-source chunks under 500 chars are semantically empty stubs | High — requires TypeScript AST sub-chunking + markdown structure awareness |
| **Query classification**          | All queries get the same retrieval pipeline; 7/20 BM25 queries and 8/20 hybrid queries score 0 MRR@5 (complete misses)                                                                                                          | Medium — local classifier or rule-based routing                            |
| **Cross-encoder re-ranking**      | Bi-encoder is fast but imprecise; cross-encoder examines query-document pairs jointly for much higher relevance                                                                                                                 | Medium — local cross-encoder model (e.g., `ms-marco-MiniLM-L-6-v2`)        |
| **Context window assembly**       | Agents receive raw ranked chunks with no deduplication, no ordering heuristics, no budget management, and no cross-chunk context headers                                                                                        | High — requires dedup, ordering, budget, and stitching pipeline            |
| **Entity/relationship graph**     | No multi-hop traversal; agents cannot discover that "Network.activate uses slab fast path" implies checking both `activate/` and `slab/` families                                                                               | High — entity extraction + lightweight graph store                         |
| **Query expansion**               | Single-query embedding misses synonymous terms, abbreviations, and domain-specific aliases                                                                                                                                      | Medium — embedding-based synonym discovery                                 |
| **Relevance feedback**            | No mechanism for agents to signal which results were useful; ranking cannot improve from interaction                                                                                                                            | Medium — feedback collection + ranking adjustment                          |
| **Structured metadata filtering** | Only `family` filter; no filtering by module boundary, export type, source path pattern, test coverage, or architectural layer                                                                                                  | Medium — metadata enrichment + filter grammar                              |
| **Multi-hop retrieval**           | Complex questions require chaining across families; current system does single-pass retrieval only                                                                                                                              | High — iterative retrieval with stopping conditions                        |
| **ANN index**                     | Native Turso DiskANN via `libsql_vector_idx` (`vector_top_k()`, `F8_BLOB` quantization) provides server-side approximate nearest neighbor search                                                                                | **Implemented** — native Turso DiskANN; no longer deferred                 |
| **RAG evaluation suite**          | Only 20 queries with MRR@5; no nDCG, recall@k, context relevance, or faithfulness metrics; no query taxonomy                                                                                                                    | Medium — eval framework + curated 50+ query set                            |

---

### Detailed audit findings

#### 1. Chunking quality — CRITICAL

**Markdown chunker (`chunker.mjs`)**:

- Default: 2,048-char max with 512-char overlap, heading-based section splitting
- **Failure mode 1 — No semantic boundary awareness**: Splits purely on heading boundaries + character count, not on content meaning. Code blocks inside markdown get split mid-function.
- **Failure mode 2 — Many chunks have no heading_path** (16,133 of 31,396, or 51.4%): Markdown sub-chunks beyond the first in a split section often get empty heading_path values, losing section context.
- **Failure mode 3 — Overlap is character-based, not sentence/paragraph-based**: The 512-char overlap window can split mid-word or mid-sentence, creating incoherent fragments at chunk boundaries.
- **Failure mode 4 — No cross-chunk context headers**: Chunks don't carry file path or parent heading chain in their body_text, so retrieval results lack provenance context.

**TypeScript AST chunker (`ts-chunker.mjs`)**:

- One chunk per exported symbol — **NO MAX SIZE LIMIT**
- **Failure mode 1 — Giant chunks**: Top chunks are 42,084 to 25,000+ chars. `all-MiniLM-L6-v2` has a 512-token max sequence length (~2,048 chars), meaning the embedding for a 42K-char chunk is computed from only the first ~2K chars of content. The remaining 40K chars are invisible to the embedding model.
- **Failure mode 2 — Tiny stubs**: 2,604 ts-source chunks are under 500 chars. These are minimal symbol declarations (e.g., re-exports, type aliases) with insufficient context for meaningful semantic matching.
- **Failure mode 3 — Duplicate heading_paths**: The symbol name becomes the heading_path, so `default` appears 15 times, `OnnxModel` 6 times, etc. Family + file_path is needed to disambiguate.
- **Failure mode 4 — No sub-chunking of large symbols**: A 42K-char utility object like `onnxImportOrchestratorsUtils` becomes one monolithic chunk, making fine-grained retrieval within it impossible.

**Chunk size distribution (ts-source):**

| Size range | Count |
| ---------- | ----- |
| under 500  | 2,604 |
| 500–1K     | 710   |
| 1K–2K      | 186   |
| 2K–5K      | 62    |
| 5K–10K     | 22    |
| 10K–20K    | 15    |
| over 20K   | 10    |

**Chunk size distribution (markdown families):**

| Size range | Count  |
| ---------- | ------ |
| under 500  | 8,348  |
| 500–1K     | 2,373  |
| 1K–2K      | 1,074  |
| over 2K    | 15,992 |

**Recommendation**: Implement two-level chunking — AST-aware splitting for TypeScript (per-symbol but with max-size sub-chunking at 1,500 chars with overlap at statement boundaries) and structure-aware markdown chunking (heading hierarchy with cross-chunk context headers). Add `parent_chunk_id`, `depth`, and `context_header` columns to the chunks table.

#### 2. Embedding model adequacy

**Measured baseline (20-query eval):**

- BM25 MRR@5: 0.225
- Hybrid MRR@5: 0.308
- Improvement: +0.083 (8.3 pp)
- 7/20 BM25 queries and 8/20 hybrid queries score 0 (complete misses)

**Model ceiling analysis:**

- `all-MiniLM-L6-v2` produces 384-dim embeddings with 512-token max sequence length
- **Primary bottleneck is NOT the model — it's chunking quality.** Giant chunks (42K chars) get truncated to 512 tokens (~2K chars), losing 95% of their content. Tiny chunks (<500 chars) lack enough context for semantic discrimination.
- Fixing chunking alone would likely improve MRR@5 more than a model upgrade.
- After chunking is fixed, upgrading to `bge-small-en-v1.5` (same 384-dim, better quality on MTEB) or `all-MiniLM-L12-v2` (384-dim, 6 more transformer layers) would provide a moderate 2-5% MRR improvement.
- The architecture MUST support model swapping. Current `embed-index.mjs` is model-agnostic (accepts `--model-id` and `--dimension`), which is good. The `model-meta.json` system allows clean model transitions.

**Recommendation**: Fix chunking first (highest impact), then evaluate `bge-small-en-v1.5` vs `all-MiniLM-L12-v2` on the expanded eval set. Design the embedding pipeline for model-hot-swapping via `model_id` + `model_sha256` in the schema.

#### 3. Ranking pipeline

**Current architecture:**

- `hybrid-rank.mjs`: `score = alpha * bm25_norm + (1 - alpha) * cosine_sim`, default alpha=0.5
- BM25 scores are min-max normalized across the candidate pool before blending
- Dense candidates are discovered via brute-force cosine similarity over all embeddings
- BM25 and dense candidate pools are merged (union) before ranking
- Candidate pool size: `max(limit * 10, 50)`

**Failure modes:**

- Fixed alpha=0.5 treats all queries identically. Short keyword queries ("NEAT crossover") would benefit from alpha=0.7 (BM25-heavy), while long natural-language queries ("how does the Flappy Bird demo use the worker pool?") need alpha=0.3 (dense-heavy).
- No cross-encoder re-ranking means the final ranking is entirely dependent on bi-encoder similarity, which is known to be imprecise for fine-grained relevance.
- No relevance feedback — agents cannot signal which results were useful.

**Recommendation**: Implement a simple query-length heuristic first (queries ≤ 3 words: alpha=0.7; queries > 6 words: alpha=0.3; else alpha=0.5), then add cross-encoder re-ranking as a second stage on the top 10-20 candidates.

#### 4. Context assembly

**Current state: NONE.** Agents receive raw ranked chunk objects with `{chunk_id, file_path, family, heading_path, text, char_start, char_end, score}`.

**Design — `assemble_context` pipeline:**

1. **Deduplication**: SHA-256 content hash (already computed for embedding integrity) + cosine similarity threshold (0.95) for near-duplicate removal
2. **Ordering heuristics**: (a) relevance score descending, (b) within same file: char_start ascending, (c) module dependency order for cross-file results
3. **Budget management**: Configurable token budget (default: 4K tokens ≈ 16K chars). When budget is exceeded, prioritize by relevance score and drop the lowest-scoring chunks first.
4. **Cross-chunk context headers**: Prefix each chunk with `[file_path > heading_path]` to give agents provenance context without inflating the embedding.
5. **Multi-family assembly**: Allow queries to span multiple families (already supported by `search_corpus`), then merge results from all families into a single ranked list.

**New MCP tool needed**: `assemble_context` — takes a query, retrieves top-K chunks, deduplicates, orders, and assembles into a token-budgeted context string with headers.

#### 5. Metadata enrichment

**Currently indexed metadata:**

- `documents`: doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at
- `chunks`: chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end
- `chunk_embeddings`: chunk_id, embedding, chunk_sha256, model_id, model_sha256, dimension, embedded_at

**Available but NOT indexed (from ts-chunker):**

- `symbol_name` — the exported TypeScript symbol name
- `signature_text` — function/class/interface signature
- `jsdoc_text` — JSDoc summary text
- `chunk_index` is always 0 for ts-source (no sub-chunking)

**Available but NOT indexed (from file paths):**

- Source path patterns (`src/architecture/network/` vs `src/neat/`)
- Module boundary (folder-based module architecture)
- Architectural layer (network, neat, methods, multithreading)
- Export type (function, class, interface, type alias, variable)

**Available but NOT indexed (from code quality scanner):**

- JSDoc quality scores (word count, presence of @param/@returns/@example)
- Cyclomatic complexity
- Test coverage status per file

**Recommendation**: Add `symbol_name`, `signature_text`, `jsdoc_text`, `export_type`, `module_path`, and `word_count` columns to the chunks table. Add a `metadata` JSON column for extensible key-value filtering. Extend `search_corpus` to accept structured filters beyond `family`.

#### 6. Multi-hop architecture

**Current state: NONE.** Single query → single retrieval pass.

**Design — iterative retrieval loop:**

- **Max 3 hops** per query (hard limit for latency budget)
- **Stopping conditions**: (a) top result score < 0.3 (diminishing returns), (b) >80% of query entities found in accumulated context, (c) no new unique chunks in hop results
- **Hop 1**: Initial query → hybrid search → top-20 candidates
- **Hop 2**: Entity extraction from Hop 1 results → expanded query with extracted entities + heading_path/file_path links → hybrid search → merge with Hop 1 results
- **Hop 3**: Result-driven refinement — identify gaps in coverage → targeted family-specific queries → merge
- **Context accumulation**: Union of all hop results, deduplicated by content hash, ordered by final relevance score

**New MCP tool needed**: `search_multi_hop` — takes a query and optional hop config, executes the iterative retrieval loop.

#### 7. Cross-encoder selection

**Candidate models for local ONNX runtime:**

| Model                                   | Size   | Latency (est.) | Quality   | Recommendation                               |
| --------------------------------------- | ------ | -------------- | --------- | -------------------------------------------- |
| `cross-encoder/ms-marco-MiniLM-L-6-v2`  | ~22MB  | ~5ms per pair  | Moderate  | **Start here** — best speed/quality tradeoff |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~33MB  | ~10ms per pair | Good      | Upgrade if L-6 is insufficient               |
| `bge-reranker-v2-m3`                    | ~568MB | ~30ms per pair | Excellent | Defer — too large for local-first constraint |

**Integration architecture**: Cross-encoder runs as a second-stage re-ranker on the top-K (10-20) hybrid candidates. This limits latency impact to ~50-200ms per query (10-20 × 10ms).

**Recommendation**: Implement `ms-marco-MiniLM-L-6-v2` as the default cross-encoder. Add `rerank` step after `rankHybridResults` in the query pipeline. Store the model alongside the bi-encoder model in `scripts/semantic-index/models/`.

#### 8. ANN index

**Current state**: Brute-force cosine similarity over all 31K embeddings per query.

**Performance analysis:**

- 31,396 chunks × 384 dims × 4 bytes/float = ~48MB per query
- Dense search latency is dominated by ONNX embedding computation (~20-50ms), not brute-force search (~5-10ms)
- At 31K scale, brute-force with caching is faster than ANN for the first ~10 queries per warm session

**Evaluation:**

| Option                              | Latency                 | Complexity | Windows CI                              | Recommendation                       |
| ----------------------------------- | ----------------------- | ---------- | --------------------------------------- | ------------------------------------ |
| Brute-force + query embedding cache | ~5-10ms search          | Low        | ✅ No native deps                       | **Current choice, keep for now**     |
| `sqlite-vec`                        | ~2-5ms search           | Medium     | ❌ Extension loading fragile on Windows | **Defer until Windows CI is stable** |
| `hnswlib-node`                      | ~1-2ms search           | Medium     | ⚠️ Native addon, needs prebuilds        | **Evaluate at >100K chunks**         |
| Brute-force + result caching        | ~1ms for cached queries | Low        | ✅                                      | **Add as incremental improvement**   |

**Recommendation**: Defer ANN until chunk count exceeds 100K. Add query-embedding caching (cache last 50 query embeddings with their results) for immediate latency wins.

#### 9. RAG eval suite

**Current state:**

- 20 queries in `eval-queries.json`
- MRR@5 only metric
- Hit criteria: `expected_doc_families` + (`expected_heading_contains` OR `expected_symbol_contains`)
- 8/20 hybrid queries score 0 (complete misses)

**Expanded eval framework design:**

- **Metrics**: MRR@5 (primary), nDCG@5, Recall@5, context relevance (human-rated on 50-query subset), faithfulness (generation quality)
- **Query taxonomy:**
  - _Simple lookup_ (8 queries): exact term match, single-family retrieval ("NEAT crossover work")
  - _Cross-boundary_ (8 queries): spans 2+ families, requires context assembly ("how does Network activation use slab?")
  - _Multi-hop_ (6 queries): requires chaining across 3+ hops ("Flappy demo worker pool integration")
  - _Exploratory_ (6 queries): broad discovery queries ("what testing approaches does the project use?")
  - _Code-specific_ (12 queries): symbol/API/function lookup ("JSDoc exported function annotation")
  - _Plan-specific_ (10 queries): roadmap/architecture/design queries ("checkpointing save resume serialization")
- **Baseline protocol**: Run current system on expanded 50+ query set, record all metrics
- **Automated eval**: Nightly CI run against expanded query set, alert on MRR@5 regression >0.01

**Recommendation**: Expand eval-queries.json from 20 to 50+ queries covering the full taxonomy. Add nDCG@5 and Recall@5 metrics. Implement a CI gate that fails on MRR@5 regression.

---

> **Source:** `plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md`, lines 121–333.
> **Extracted:** 2026-06-08. Content preserved verbatim for permanent reference.
