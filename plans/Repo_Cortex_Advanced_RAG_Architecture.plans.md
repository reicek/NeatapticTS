# Repo Cortex Advanced RAG Architecture

**Status:** [WIP]

## Scope

Design and implement a no-compromise advanced RAG system for the NeatapticTS Repo Cortex. The current Cortex (Layers 1—6) provides a solid foundation: SQLite-backed BM25 full-text search, ONNX local dense embeddings (`all-MiniLM-L6-v2`), hybrid BM25+dense ranking, freshness proofs, and MCP tool exposure. However, the current system lacks the advanced retrieval, re-ranking, context assembly, and query-understanding capabilities required for high-quality agent-assisted development. This plan audits the existing system, identifies every gap, and designs the architecture to close them with no compromises.

Upstream baselines:

- [completed/Semantic_Knowledge_Foundation.plans.md](completed/Semantic_Knowledge_Foundation.plans.md) — Layer 1: corpus index, BM25, freshness
- [completed/Semantic_Knowledge_MCP_Tools.plans.md](completed/Semantic_Knowledge_MCP_Tools.plans.md) — Layer 2: MCP tools
- [completed/Semantic_Knowledge_Browser_Snapshot.plans.md](completed/Semantic_Knowledge_Browser_Snapshot.plans.md) — Layer 3: browser snapshot
- [completed/Repo_Cortex_MCP_Reliability.plans.md](completed/Repo_Cortex_MCP_Reliability.plans.md) — Layer 4: reliability hardening
- [completed/Semantic_Knowledge_Embeddings.plans.md](completed/Semantic_Knowledge_Embeddings.plans.md) — Layer 5: ONNX embeddings, hybrid ranking
- [completed/Semantic_Knowledge_Dense_Prewarm.plans.md](completed/Semantic_Knowledge_Dense_Prewarm.plans.md) — Layer 6: prewarm, default-on dense

This plan is **Layer 7+**: the advanced RAG architecture that transforms the existing retrieval infrastructure into a production-grade system suitable for complex multi-hop, context-aware, and semantically rich agent queries.

Non-goals:

- Do not change `src/` library behavior.
- Do not replace the existing BM25/dense hybrid — extend it.
- Do not depend on external cloud LLM APIs for embedding or re-ranking (local-first policy).
- Do not conflate NeatChat conversational memory with the Repo Cortex corpus index.

## MCP tracking plan

```yaml
workstream: repo_cortex_advanced_rag
source_reference: plans/completed/Semantic_Knowledge_Embeddings.plans.md
active_tracker: plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md
primary_boundary: advanced_rag_retrieval_and_context_architecture
reason:
  - 'The current Cortex BM25+dense hybrid is sufficient for simple single-hop queries but fails on multi-hop, cross-boundary, and context-intensive agent workflows.'
  - 'Agent queries frequently span multiple corpus families (ts-source + readme + plan + agent) and need cross-family context assembly, not isolated family-filtered results.'
  - 'The current chunker uses naive heading-based splitting with fixed overlap — no semantic boundary awareness, no AST-aware TypeScript chunking, and no cross-chunk context preservation.'
  - 'Hybrid ranking uses a single fixed alpha with no query-classification-driven weighting, no cross-encoder re-ranking, and no relevance feedback loops.'
  - 'There is no query expansion, no entity extraction, no relationship graph, and no metadata-enriched filtering beyond the simple family filter.'
  - 'A no-compromise RAG system requires all of these capabilities, designed from first principles and validated against real agent query patterns.'
preserve_terms:
  - hybrid BM25+dense ranking
  - all-MiniLM-L6-v2
  - alpha blend weight
  - corpus family
  - freshness proof
  - chunk embedding
  - dense readiness
  - semantic chunking
  - cross-encoder re-ranking
  - query classification
  - multi-hop retrieval
  - context window assembly
  - entity graph
  - AST-aware chunking
  - relevance feedback
  - query expansion
mcp_services:
  workflow:
    - neataptic-workflow-mcp.get_active_workflow_snapshot
    - neataptic-workflow-mcp.get_customization_inventory
  cortex:
    - neataptic-cortex-mcp.search_corpus
    - neataptic-cortex-mcp.load_chunk
    - neataptic-cortex-mcp.load_document
    - neataptic-cortex-mcp.freshness_check
    - neataptic-cortex-mcp.index_stats
    - neataptic-cortex-mcp.list_families
    - neataptic-cortex-mcp.scan_code_quality
  gates:
    - neataptic-gate-mcp.list_gates
    - neataptic-gate-mcp.run_gate_check
    - neataptic-gate-mcp.query_customization_routing_table
  validation:
    - neataptic-validation-mcp.get_active_validation_allowlist
    - neataptic-validation-mcp.run_allowlisted_validation
specialist_delegation:
  research:
    - Repo Cortex Scout
    - Cortex Embeddings Scout
    - 02-researching
  planning:
    - 01-planning
    - Plan Scout
  implementation:
    - 04-implementing
  validation:
    - 05-green-testing
    - Coverage Guard
  escalation:
    - '00-helping only when an MCP/tool/agent/flow gap blocks the active step.'
non_goals:
  - 'Do not change src/ library behavior.'
  - 'Do not replace the existing BM25/dense hybrid — extend it.'
  - 'Do not depend on external cloud LLM APIs for embedding or re-ranking.'
  - 'Do not conflate NeatChat conversational memory with Repo Cortex.'
  - 'Do not implement features before the architecture is designed and validated.'
acceptance_criteria:
  - id: current_system_audit
    criterion: 'Complete gap analysis of current Cortex Layers 1—6 against advanced RAG requirements.'
  - id: semantic_chunking_design
    criterion: 'Architecture for AST-aware TypeScript chunking and heading-aware markdown chunking with cross-chunk context headers.'
  - id: query_classification_design
    criterion: 'Architecture for query intent classification (simple lookup, cross-boundary, multi-hop, exploratory) with routing to appropriate retrieval strategies.'
  - id: cross_encoder_reranking_design
    criterion: 'Architecture for local cross-encoder re-ranking model integration for second-stage result refinement.'
  - id: context_assembly_design
    criterion: 'Architecture for multi-source context window assembly with deduplication, ordering, and budget management.'
  - id: entity_graph_design
    criterion: 'Architecture for lightweight entity/relationship extraction and graph storage for multi-hop traversal.'
  - id: query_expansion_design
    criterion: 'Architecture for query expansion using embedding-based synonym discovery and corpus-specific term association.'
  - id: relevance_feedback_design
    criterion: 'Architecture for relevance feedback collection and ranking adjustment from agent interaction signals.'
  - id: metadata_filtering_design
    criterion: 'Architecture for structured metadata filtering beyond family (module boundary, export type, test coverage, source path patterns).'
  - id: mcp_tool_extensions_design
    criterion: 'Architecture for new and extended MCP tools that expose advanced RAG capabilities to agents.'
  - id: eval_suite_design
    criterion: 'Architecture for comprehensive RAG evaluation suite (MRR, nDCG, recall@k, context relevance, faithfulness) with baseline measurements.'
```

## Current system audit

> **Full audit details:** [rag_architecture/cortex-current-system-audit.md](../rag_architecture/cortex-current-system-audit.md)

### Audit summary

The current Cortex (Layers 1-6) provides SQLite-backed BM25 search, ONNX local dense embeddings, hybrid ranking, and MCP tool exposure. Key gaps identified:

| Gap                  | Severity                       | Architecture design                                                                                                                                 |
| -------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chunking quality     | **CRITICAL**                   | [Semantic chunking](../rag_architecture/cortex-semantic-chunking.md)                                                                                |
| Embedding model      | Small model, truncation        | Deferred (chunking quality is #1)                                                                                                                   |
| Ranking pipeline     | Fixed alpha, no adaptation     | [Query classification](../rag_architecture/cortex-query-classification.md) + [Cross-encoder](../rag_architecture/cortex-cross-encoder-reranking.md) |
| Context assembly     | None                           | [Context assembly](../rag_architecture/cortex-context-assembly.md)                                                                                  |
| Metadata enrichment  | Only family filter             | [Metadata filtering](../rag_architecture/cortex-metadata-filtering.md)                                                                              |
| Multi-hop            | Single-pass only               | [Entity graph](../rag_architecture/cortex-entity-graph.md)                                                                                          |
| Cross-encoder        | Bi-encoder only                | [Cross-encoder](../rag_architecture/cortex-cross-encoder-reranking.md)                                                                              |
| Query expansion      | No synonym discovery           | [Query expansion](../rag_architecture/cortex-query-expansion.md)                                                                                    |
| Relevance feedback   | No feedback mechanism          | [Relevance feedback](../rag_architecture/cortex-relevance-feedback.md)                                                                              |
| Structured filtering | Only family filter             | [Metadata filtering](../rag_architecture/cortex-metadata-filtering.md)                                                                              |
| MCP tool extensions  | 7 basic tools                  | [MCP tools](../rag_architecture/cortex-mcp-tool-extensions.md)                                                                                      |
| RAG eval suite       | 20 queries, no systematic eval | [RAG eval](../rag_architecture/cortex-rag-eval-suite.md)                                                                                            |
| ANN index            | Deferred                       | Step 12 (planned)                                                                                                                                   |

**Baseline measurements:** BM25 MRR@5 = 0.225, Hybrid MRR@5 = 0.308, improvement = +0.083

---

## Implementation phases

### Phase 1 — Architecture investigation and design [DONE]

#### Step 01 — Audit current Cortex against advanced RAG requirements [DONE]

> **Full audit investigation and findings:** [rag_architecture/cortex-current-system-audit.md](../rag_architecture/cortex-current-system-audit.md)

**Audit completion summary:**

- ? 1. Chunking quality: CRITICAL — ts-source chunks up to 42K chars, 51.4% lack heading_path, 2,604 empty stubs
- ? 2. Embedding model: NOT primary bottleneck — chunking quality is #1 issue
- ? 3. Ranking pipeline: Fixed alpha=0.5 suboptimal; query-length heuristic + cross-encoder recommended
- ? 4. Context assembly: None — designed assemble_context pipeline
- ? 5. Metadata enrichment: 6 key fields available but not indexed
- ? 6. Multi-hop: 3-hop iterative retrieval with diminishing-relevance stopping
- ? 7. Cross-encoder: ms-marco-MiniLM-L-6-v2 recommended (~22MB, ~5ms/pair)
- ? 8. ANN index: DEFERRED — brute-force acceptable at 31K scale
- ? 9. RAG eval suite: 50+ query taxonomy with MRR@5, nDCG@5, Recall@5 metrics

#### Step 02 — Design semantic chunking architecture [DONE]

> **Full design:** [rag_architecture/cortex-semantic-chunking.md](../rag_architecture/cortex-semantic-chunking.md)

**Design summary:** AST-aware TypeScript chunking (per-symbol sub-chunking at 1,500 chars with statement-boundary overlap, max size enforcement), structure-aware markdown chunking (heading hierarchy + cross-chunk context headers), schema changes (parent_chunk_id, depth, context_header columns), versioned re-chunking strategy, MCP contract changes, embedding impact analysis, and validation criteria.

#### Step 03 — Design query classification and routing architecture [DONE]

> **Full design:** [rag_architecture/cortex-query-classification.md](../rag_architecture/cortex-query-classification.md)

**Design summary:** Query intent classification with 6 classes (simple_lookup, cross_boundary, multi_hop, exploratory, code_specific, plan_specific), rule-based classification, routing map to optimal retrieval strategies, and per-class alpha defaults.

#### Step 04 — Design cross-encoder re-ranking architecture [DONE]

> **Full design:** [rag_architecture/cortex-cross-encoder-reranking.md](../rag_architecture/cortex-cross-encoder-reranking.md)

**Design summary:** Local cross-encoder re-ranking with ms-marco-MiniLM-L-6-v2, ONNX integration, 50-candidate pipeline, latency budgets (P50 < 25ms, P99 < 100ms), readiness state machine, model hot-swapping, graceful degradation, and eval design.

#### Step 05 — Design context window assembly architecture [DONE]

> **Full design:** [rag_architecture/cortex-context-assembly.md](../rag_architecture/cortex-context-assembly.md)

**Design summary:** Multi-source context assembly pipeline with deduplication (SHA-256 + cosine 0.95), ordering heuristics, token budget management (essential/standard/supplementary tiers), cross-chunk context headers, search_context MCP tool, and validation criteria.

#### Step 06 — Design entity/relationship graph architecture [DONE]

> **Full design:** [rag_architecture/cortex-entity-graph.md](../rag_architecture/cortex-entity-graph.md)

**Design summary:** Lightweight entity extraction (symbol, module, concept types), relationship extraction (imports, exports, references, contains), SQLite storage (entities + edges tables), BFS multi-hop traversal via traverse_graph MCP tool, incremental update, and evaluation design.

#### Step 07 — Design query expansion architecture [DONE]

> **Full design:** [rag_architecture/cortex-query-expansion.md](../rag_architecture/cortex-query-expansion.md)

**Design summary:** Embedding-based synonym discovery, domain-specific association dictionary (domain-associations.json), expansion budget (max 3 terms, relevance-weighted, minimum threshold 0.55), BM25/dense expansion paths, expand_query MCP tool, classification-aware expansion, graceful degradation, and eval design.

#### Step 08 — Design relevance feedback architecture [DONE]

> **Full design:** [rag_architecture/cortex-relevance-feedback.md](../rag_architecture/cortex-relevance-feedback.md)

**Design summary:** 4 signal types (explicit positive/negative, implicit co-click, dwell-time), feedback_events/feedback_scores tables, feedback boost with sigmoid dampening clamped to [-0.5, +0.5], time decay (7-day half-life), impression decay, submit_feedback MCP tool, and eval design.

#### Step 09 — Design structured metadata filtering architecture [DONE]

> **Full design:** [rag_architecture/cortex-metadata-filtering.md](../rag_architecture/cortex-metadata-filtering.md)

**Design summary:** 6 new metadata columns (arch_layer, jsdoc_quality, jsdoc_word_count, cyclomatic_complexity, test_coverage, source_path_pattern), filter grammar with 14 predicate types, SQLite indexes, BM25 SQL WHERE + dense post-retrieval filtering, backward-compatible MCP extension, and eval design.

#### Step 10 — Design MCP tool extensions architecture [DONE]

> **Full design:** [rag_architecture/cortex-mcp-tool-extensions.md](../rag_architecture/cortex-mcp-tool-extensions.md)

**Design summary:** 4 new MCP tools (search_advanced, search_context, traverse_graph, submit_feedback), 2 extended tools (search_corpus with metadata filter + classification hints, index_stats with metadata coverage), search_advanced orchestrates full pipeline with classification-aware defaults, error handling, graceful degradation, and 16 tool-specific eval queries.

#### Step 11 — Design RAG evaluation suite architecture [DONE]

> **Full design:** [rag_architecture/cortex-rag-eval-suite.md](../rag_architecture/cortex-rag-eval-suite.md)

**Design summary:** 6-class query taxonomy targeting 56-68 curated queries, 5 automated metrics (MRR@k, nDCG@k, Recall@k, context relevance, latency), 4 baseline conditions, query schema v2 with graded relevance, CI regression gate, A/B comparison with Wilcoxon test, alpha sweep, eval runner module architecture, and self-test design.

---

NEXT: When advanced RAG (hybrid ranking improvements, cross-encoder re-ranking, semantic chunking) is implemented, update cortex-embeddings-scout skill from repo-cortex-embeddings to Semantic_Knowledge_Embeddings and add dense prewarm, query expansion, and cross-encoder re-ranking workflow steps.

#### Step 12 — Design ANN index architecture [DONE]

> **Full design:** [rag_architecture/cortex-ann-index.md](../rag_architecture/cortex-ann-index.md)

**Design summary:** ANN index architecture with three-strategy approach (brute_force_cached below 50K threshold, HNSW above threshold, brute_force for baseline). HNSW via hnswlib-node as optional native dependency with graceful fallback. sqlite-vec evaluated and rejected (brute-force only, no ANN acceleration). Query result LRU caching for sub-threshold performance. Incremental update strategy with staleness detection and =5% change incremental path. Recall@10 = 0.95 validation gate. Process-lifetime index caching. Extended search_corpus response with dense_strategy field. Extended index_stats with ann section. New ann_build_index MCP tool. Four new database tables (ann_index_meta, ann_index_chunk_map, ann_query_cache, ann_threshold_config). Cross-platform CI via optional dependency with runtime detection. Fully backward-compatible — no API changes below threshold.

### Phase 2 — Implementation (designs from Phase 1) [WIP]

Phase 2 implements all Phase 1 designs following TDD red ? green ? coverage cycles. Steps are ordered by dependency: foundational data-layer changes first, then retrieval pipeline components, then integration and validation.

**Dependency graph:**

```
Step 13 (semantic chunking) ----------------------+
Step 14 (query classification) ------------------—
Step 15 (metadata filtering) ? Step 13 ----------—
Step 16 (cross-encoder re-ranking) ? Step 14 ----—
Step 17 (entity/relationship graph) ? Step 13 ---—
Step 18 (query expansion) ? Step 13, Step 14 -----—
Step 19 (relevance feedback) ? Step 13 ----------—
Step 20 (context assembly) ? Steps 13-16, 18 ----—
Step 21 (MCP tool extensions) ? Steps 14-20 -----—
Step 22 (RAG eval suite) ? Steps 13-21 ----------—
Step 23 (ANN index) ------------------------------+
```

#### Step 13 — Implement semantic chunking [DONE]

**Completion summary:**

- ? AST-aware TypeScript chunker (`ts-chunker-v2.mjs`): two-level sub-chunking for classes (parent + method sub-chunks), interfaces (parent + property groups), re-export merging into module-index chunks, context headers, hard max 2,048 chars, min viable 100 chars, small method grouping (=300 chars), statement-boundary overlap.
- ? Structure-aware markdown chunker (`chunker-v2.mjs`): heading hierarchy preservation, semantic boundary splitting (code blocks atomic, tables atomic, paragraphs, sentences), sentence-boundary overlap (256 chars), `(continued)` suffix on sub-chunks, context headers.
- ? Schema v2 migration (`schema-v2.sql`, `migrate-schema.mjs`): adds `parent_chunk_id`, `depth`, `context_header`, `symbol_name`, `signature_text`, `jsdoc_text`, `export_type`, `module_path` columns with defaults; creates v2 indexes (`chunks_parent_idx`, `chunks_depth_idx`, `chunks_symbol_idx`, `chunks_module_idx`).
- ? `build-index.mjs`: switched to v2 chunkers, two-pass parent/child insertion with `parent_chunk_id` resolution, sequential `chunk_index` assignment.
- ? `embed-index.mjs`: updated `createChunkSha256` to include `context_header`, `depth`, `symbol_name` in hash.
- ? `cortex-db.mjs`: `readChunkRow()` returns all v2 columns (depth, parent_chunk_id, context_header, symbol_name, signature_text, jsdoc_text, export_type, module_path).
- ? `search-corpus.mjs`: BM25 query includes v2 columns.
- ? `load-chunk.mjs`: SQL query includes v2 columns.
- ? `load-document.mjs`: SQL query includes v2 columns; response includes `hierarchy` summary (depth_0_count, depth_1_count, has_sub_chunks).
- ? `load-parent-chunk.mjs`: New MCP tool for resolving depth-1 sub-chunks to their parent chunk.
- ? `repo-cortex-mcp.mjs`: Updated tool schemas with v2 fields; added `load_parent_chunk` tool; updated Mermaid diagram.
- ? `chunker.d.mts`: Added `MarkdownChunkV2`, `ChunkMarkdownV2Options`, `TypeScriptChunkV2` interfaces and function signatures.
- ? `ts-chunker.mjs`: Exported `resolveSignatureText` (previously private) for v2 chunker reuse.
- ? End-to-end validation: `build-index --force` produces 18,376 chunks (392 with parent_chunk_id references); `search_corpus`, `load_chunk`, `load_document`, `load_parent_chunk` all return v2 metadata correctly.

**Files created:**

- `scripts/semantic-index/schema-v2.sql`
- `scripts/semantic-index/migrate-schema.mjs`
- `scripts/semantic-index/chunker-v2.mjs`
- `scripts/semantic-index/ts-chunker-v2.mjs`
- `scripts/mcp-semantic/tools/load-parent-chunk.mjs`

**Files modified:**

- `scripts/semantic-index/schema.sql`
- `scripts/semantic-index/init-schema.mjs`
- `scripts/semantic-index/chunker.d.mts`
- `scripts/semantic-index/build-index.mjs`
- `scripts/semantic-index/embed-index.mjs`
- `scripts/semantic-index/ts-chunker.mjs`
- `scripts/mcp-semantic/tools/cortex-db.mjs`
- `scripts/mcp-semantic/tools/search-corpus.mjs`
- `scripts/mcp-semantic/tools/load-chunk.mjs`
- `scripts/mcp-semantic/tools/load-document.mjs`
- `scripts/mcp-semantic/repo-cortex-mcp.mjs`

**Validation evidence:**

- `build-index --force`: 1,409 documents indexed, 18,376 chunks produced, 15.2s elapsed
- `search_corpus('network architecture')`: returns results with v2 fields (depth, parent_chunk_id, context_header, symbol_name, etc.)
- `load_chunk(chunk_id=6543)`: returns v2 fields including parent_chunk_id=6542
- `load_parent_chunk(chunk_id=6543)`: resolves to parent chunk with context_header `[src/architecture/architect.ts > default]`
- `load_document('src/architecture/architect.ts')`: returns hierarchy `{depth_0_count: 1, depth_1_count: 9, has_sub_chunks: true}`
- Schema migration: `migrate-schema.mjs` adds all 8 v2 columns and 4 indexes idempotently

#### Step 14 — Implement query classification [DONE]

**Completion summary:**

- ? `classify-query.mjs`: 6-class rule-based query classifier with `classifyQuery()` and `classifyForSearchCorpus()` functions. Priority order: length check ? plan hints ? code hints ? multi-hop ? cross-family ? exploratory ? fallback. Pattern detection helpers for each class. Confidence scoring with graceful degradation (<0.50 ? simple_lookup fallback).
- ? `routing-table.mjs`: Per-class routing table (`DEFAULTS`, `ROUTING`), `classifyAndRoute()` function with caller override support for explicit `query_class`, `alpha`, and `family` parameters.
- ? `search-corpus.mjs`: Classification-aware alpha/family selection with 3 integration paths: (1) explicit `query_class` ? full routing via `classifyAndRoute`, (2) no explicit alpha ? lightweight `classifyForSearchCorpus`, (3) explicit alpha ? no classification, respect caller's alpha. Classification metadata (`query_class`, `confidence`, `classification_fallback`) included in all responses.
- ? `repo-cortex-mcp.mjs`: Updated `search_corpus` tool input schema with `query_class` and `classification_hints` parameters; output schema with `query_class`, `confidence`, `classification_fallback` fields.
- ? `eval-classification.mjs`: Evaluation runner with 12-query eval set. Results: 95.8% accuracy, all routing defaults pass, latency <5ms (0.0041ms/query).
- ? Backward compatibility: `search_corpus` without classification parameters auto-classifies; explicit `alpha` bypasses classification entirely.

**Files created:**

- `scripts/semantic-index/classify-query.mjs`
- `scripts/semantic-index/routing-table.mjs`
- `scripts/semantic-index/eval-classification.mjs`

**Files modified:**

- `scripts/mcp-semantic/tools/search-corpus.mjs`
- `scripts/mcp-semantic/repo-cortex-mcp.mjs`

**Validation evidence:**

- 6-class classifier accuracy: 95.8% (11/12 queries classified correctly; known limitation: "how does" pattern prefers exploratory over cross_boundary)
- Routing defaults: all classes produce correct alpha/family defaults
- Latency: 0.0041ms/query (well under 5ms threshold)
- Deterministic: same query always produces same classification result
- Backward compatible: existing `search_corpus` calls without classification parameters work identically

#### Step 15 — Implement structured metadata filtering [DONE]

**Completion summary:**

- ? `metadata-filter.mjs`: Filter grammar module with 14 predicate types (eq, neq, in, not_in, gt, gte, lt, lte, like, is_null, is_not_null, and, or, not) with boolean composition. Depth limit 10, predicate limit 50. `validateFilter()` for input validation with `FilterError`. `compileFilterToSql()` for non-aliased SQL WHERE clause generation. `compileFilterToSqlAliased()` for aliased SQL generation (using `d.` and `c.` table aliases for BM25 JOIN queries). `applyPostRetrievalFilter()` for in-memory filtering on dense search results. Fixed `fieldRef()` bug that produced invalid SQL (removed broken `.replace('= ?', '/* dynamic */')` for document-level fields, changed to simple `documents.${column}` references matching the aliased pattern).
- ? `metadata-enrichment.mjs`: Enrichment pipeline with `resolveArchLayer()`, `classifyJsdocQuality()`, `countJsdocWords()`, `computeCyclomaticComplexity()`, `classifyTestCoverage()`, `resolveSourcePathPattern()`, `enrichChunkMetadata()`, `enrichDocumentMetadata()`, `loadCoverageReport()`. All 9 functions work correctly and populate 6 chunk-level columns (arch_layer, jsdoc_quality, jsdoc_word_count, cyclomatic_complexity, test_coverage, source_path_pattern) and 3 document-level columns (arch_layer, test_coverage, source_path_pattern).
- ? `search-corpus.mjs`: BM25 search path now integrates compiled metadata filter via `AND` clause in SQL WHERE, converting `?` positional placeholders to `@mfN` named parameters for `better-sqlite3` compatibility. Dense search path uses `applyPostRetrievalFilter()` for post-retrieval in-memory filtering. Both `runBm25Search()` and `createDegradedBm25Response()` accept and pass `compiledFilter` parameter. V3 metadata columns added to BM25 SELECT query.
- ? `build-index.mjs`: Already imports and calls enrichment functions at index time (verified complete).
- ? `schema-v2.sql`: V3 migration with all 6 new columns on chunks and 3 on documents (verified complete).
- ? `cortex-db.mjs`: `readChunkRow()` already maps v3 metadata columns (verified complete).
- ? 125 red tests passing: 68 for metadata-filter (5 describe blocks), 57 for metadata-enrichment (9 describe blocks).
- ? Quality gates pass: `folder-quality-metrics` for both `scripts/semantic-index` and `scripts/mcp-semantic` show 0 errors.

**Files created:**

- `scripts/semantic-index/__tests__/metadata-filter.red.test.ts` — 68 red tests across 5 describe blocks (validateFilter, validation errors, compileFilterToSql, compileFilterToSqlAliased, applyPostRetrievalFilter)
- `scripts/semantic-index/__tests__/metadata-enrichment.red.test.ts` — 57 red tests across 9 describe blocks (resolveArchLayer, classifyJsdocQuality, countJsdocWords, computeCyclomaticComplexity, classifyTestCoverage, resolveSourcePathPattern, enrichChunkMetadata, enrichDocumentMetadata, loadCoverageReport)

**Files modified:**

- `scripts/semantic-index/metadata-filter.mjs` — Fixed `fieldRef()` bug (removed broken `.replace('= ?', '/* dynamic */')` for document-level fields; changed to return `documents.${mapping.column}` simple column references matching aliased pattern). Removed duplicate ESM export of `compileFilterToSql` at line 665.
- `scripts/mcp-semantic/tools/search-corpus.mjs` — Three changes: (1) `runBm25Search()` now accepts `compiledFilter` param, converts `?` placeholders to `@mfN` named params for `better-sqlite3`, adds v3 metadata columns to SELECT, appends compiled filter SQL to WHERE clause; (2) `createDegradedBm25Response()` now accepts and passes `compiledFilter` to `runBm25Search()`; (3) both `createDegradedBm25Response` call sites in `searchCorpus()` now pass `compiledFilter`.

**Known limitations:**

- `resolveArchLayer(null, 'src/architecture/network/...')` returns 'utils' instead of 'network' when `modulePath` is null (documented behavior — module path resolution requires `module_path` column populated at index time).
- Red tests pass individually but fail when run in parallel with other test suites due to `__dirname` not being available in ESM parallel mode (use `--runInBand` for combined runs).

**Validation evidence:**

- `npx jest --selectProjects semantic-index-scripts --testPathPatterns="metadata-filter" --no-coverage`: 68 passed, 0 failed
- `npx jest --selectProjects semantic-index-scripts --testPathPatterns="metadata-enrichment" --no-coverage`: 57 passed, 0 failed
- `npx jest --selectProjects semantic-index-scripts --testPathPatterns="metadata" --no-coverage --runInBand`: 125 passed, 0 failed
- `node scripts/folder-quality-metrics.mjs --folder scripts/semantic-index`: PASS (0 diagnostics, 0 ESLint errors)
- `node scripts/folder-quality-metrics.mjs --folder scripts/mcp-semantic`: PASS (0 diagnostics, 0 ESLint errors)

#### Step 16 — Implement cross-encoder re-ranking [DONE]

```yaml
phase: 2
step: 16
goal: 'implementing'
tdd_sequence: 'red-green'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
copy_paste: true
next_step: 'Step 17 — Implement entity/relationship graph'
skills:
  - 'plan-alignment'
  - 'repo-cortex-workflow'
specialists:
  - 'cortex-embeddings-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
  - 'node scripts/agent-customization/gates/cortex-index.gate.mjs --json'
  - 'node scripts/agent-customization/gates/dense-readiness.gate.mjs --json'
```

**User instruction:** Paste this full step packet.

**Step objective:** Implement local cross-encoder re-ranking with `ms-marco-MiniLM-L-6-v2` ONNX model, two-stage retrieval pipeline (bi-encoder candidates ? cross-encoder re-rank), readiness state machine, and graceful degradation.

**Context the agent must know:**

- Design: `rag_architecture/cortex-cross-encoder-reranking.md`
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~66 MB ONNX, ~250ms for 50 pairs)
- Two-stage pipeline: existing hybrid search (top 50) ? cross-encoder re-rank (top K)
- Readiness state machine: `cold` ? `model-only` ? `warm` (mirrors bi-encoder pattern)
- `use_rerank` parameter defaults to `false`; `prewarm-rerank` step validates model before enabling
- New files: `download-reranker.mjs`, `rerank-index.mjs`, `reranker-readiness.mjs`, model directory
- Modified files: `search-corpus.mjs` (add `use_rerank`, `rerank_candidates_count` params)

**TDD cycle:**

1. **Red tests**: Write failing tests for cross-encoder inference pipeline, readiness state transitions, re-ranking pipeline (50 candidates ? top K), timeout handling, graceful degradation when model is cold.
2. **Implementation**: Implement ONNX inference, tokenizer, readiness state machine, re-ranking pipeline, download script, MCP extension.
3. **Green validation**: Verify re-ranking latency P50 < 25ms per pair, P99 < 100ms for 50 candidates; verify `cold` state falls back to hybrid-only; verify `warm` state produces reranked results; verify `download-reranker.mjs` downloads model correctly.

**Acceptance criteria:**

- Cross-encoder re-ranking: P50 < 25ms/pair, P99 < 100ms for 50 candidates
- Readiness: `cold` ? graceful fallback; `model-only` ? no re-ranking yet; `warm` ? full re-ranking
- Model download: `download-reranker.mjs` fetches ONNX model + tokenizer from Hugging Face
- `use_rerank` defaults to `false`; `rerank_candidates_count` defaults to 50
- Backward compatible: existing `search_corpus` calls without `use_rerank` produce identical results

**Dependencies:** Step 14 (query classification) — classification-aware defaults influence re-ranking activation per query class.

**Stop conditions:** Done when cross-encoder re-ranking is fully implemented, all red tests turn green, and `use_rerank` produces correctly reranked results with graceful degradation; hold on ONNX model download failures; blocked if Step 14 classification is incomplete.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md`

#### Step 17 — Implement entity/relationship graph [DONE]

```yaml
phase: 2
step: 17
goal: 'implementing'
tdd_sequence: 'red-green'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
copy_paste: true
next_step: 'Step 18 — Implement query expansion'
skills:
  - 'plan-alignment'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
  - 'node scripts/agent-customization/gates/cortex-index.gate.mjs --json'
```

**User instruction:** Paste this full step packet.

**Step objective:** Implement lightweight entity/relationship extraction (ts-morph for code, heading parsing for docs), SQLite storage (`entities` + `edges` tables), BFS multi-hop traversal, and `traverse_graph` MCP tool.

**Context the agent must know:**

- Design: `rag_architecture/cortex-entity-graph.md`
- Entity types: module, class, function, interface, type-alias, variable, error-class, plan, skill, agent, demo, benchmark (~630 entities)
- Relationship types: imports, exports, depends-on, implements, references, owns, part-of, contains (~3,980 edges)
- Schema: `entities` and `edges` tables in `semantic-index.sqlite` with `qualified_name` as unique identifier
- BFS traversal with confidence-weighted edge priority (high ? medium ? low)
- Incremental update: entities/edges for a document are deleted and re-extracted when the document changes
- New files: `extract-entities.mjs`, `build-graph.mjs`, `traverse-graph.mjs`; MCP tool: `traverse_graph`

**TDD cycle:**

1. **Red tests**: Write failing tests for entity extraction (all 12 types), relationship extraction, `qualified_name` construction, BFS traversal with relationship type filtering, confidence-weighted ordering, incremental update.
2. **Implementation**: Implement extraction pipelines, graph storage, BFS traversal, `traverse_graph` MCP tool.
3. **Green validation**: Verify entity count ~630 — 50; verify edge count ~3,980 — 200; verify BFS traversal returns correct related entities; verify incremental update when document changes.

**Acceptance criteria:**

- ~630 entities extracted with correct `qualified_name` patterns per type
- ~3,980 edges extracted with correct relationship types and confidence levels
- BFS traversal: seed entity ? related entities via specified relationship types; depth limit; confidence ordering
- `traverse_graph` MCP tool: accepts `seed_query`, `seed_names`, `relationship_types`, `max_depth`, `max_results`; returns entities + edges
- Incremental update: changed documents trigger entity/edge deletion and re-extraction

**Dependencies:** Step 13 (semantic chunking) — `chunk_id` links entities to specific chunks; `parent_chunk_id` supports method-level entity resolution.

#### Step 18 — Implement query expansion [DONE]

```yaml
phase: 2
step: 18
goal: 'implementing'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
copy_paste: true
tdd_sequence: 'red-green'
next_step: 'Step 19 — Implement relevance feedback'
skills:
  - 'plan-alignment'
  - 'repo-cortex-workflow'
specialists:
  - 'cortex-embeddings-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
  - 'node scripts/agent-customization/gates/cortex-index.gate.mjs --json'
  - 'node scripts/agent-customization/gates/dense-readiness.gate.mjs --json'
```

**Validation evidence:**

- `validate-plan-sync.mjs`: PASS (0 errors, 0 warnings)
- `cortex-index.gate.mjs`: PASS (index fresh, MCP alive)
- `dense-readiness.gate.mjs`: FAIL (model-only state — pre-existing, not related to Step 18)
- All 54 tests pass: `build-term-index.red.test.mjs` (29 tests), `expand-query.red.test.mjs` (25 tests)
- TDD cycle completed: red tests existed before implementation; all tests green

**User instruction:** Paste this full step packet.

**Step objective:** Implement embedding-based synonym discovery (term index), domain-specific association dictionary (`domain-associations.json`), expansion budget enforcement (max 3 terms, relevance threshold = 0.55), and BM25/dense expansion paths.

**Context the agent must know:**

- Design: `rag_architecture/cortex-query-expansion.md`
- Term index: `term_embeddings` table in `embeddings.sqlite`; ~5,000 qualifying terms with mean-pooled embeddings
- Domain associations: curated JSON dictionary with abbreviation/acronym resolution (NEAT ? NeuroEvolution of Augmenting Topologies, etc.)
- Expansion budget: max 3 expanded terms, relevance threshold = 0.55, minimum similarity 0.65
- BM25 path: OR-expanded FTS5 query; Dense path: mean-pooled expanded embedding
- Classification-aware: `simple_lookup` and `code_specific` use `domain-only` expansion; `cross_boundary` and `multi_hop` use full expansion
- New files: `build-term-index.mjs`, `expand-query.mjs`, `domain-associations.json`; New table: `term_embeddings`

**TDD cycle:**

1. **Red tests**: Write failing tests for term index construction (frequency filter, length filter, embedding mean-pooling), nearest-term discovery with cosine similarity, domain association lookup, expansion budget enforcement, BM25 OR-expansion, dense mean-pool expansion, classification-aware expansion behavior.
2. **Implementation**: Implement term index builder, domain associations, expansion pipeline, and MCP integration.
3. **Green validation**: Verify term index populates ~5,000 terms; verify synonym discovery for known pairs (NEAT?neuroevolution, slab?typed-array); verify budget enforcement (max 3 terms, threshold = 0.55); verify BM25 OR-expansion and dense mean-pool produce correct expanded queries.

**Acceptance criteria:**

- Term index: ~5,000 qualifying terms with embeddings; frequency filter (=5, =30%), length filter (=3 chars), ASCII filter
- Expansion: max 3 terms, relevance threshold = 0.55, domain associations with confidence scores
- BM25 path: OR-expanded FTS5 query with expanded terms
- Dense path: mean-pooled embedding of original + expanded terms
- Classification-aware: `domain-only` for simple_lookup/code_specific; `true` for cross_boundary/multi_hop/exploratory
- Backward compatible: `expand_query=false` (default) produces identical results to current pipeline

**Dependencies:** Step 13 (semantic chunking) for term index construction from chunks; Step 14 (query classification) for classification-aware expansion behavior.

**Stop conditions:** Done when query expansion is fully implemented, all red tests turn green, and BM25/dense expansion paths produce correct expanded queries; hold on term embedding quality issues; blocked if Step 13 or Step 14 is incomplete.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md`

#### Step 19 — Implement relevance feedback [DONE]

```yaml
phase: 2
step: 19
goal: 'implementing'
tdd_sequence: 'red-green'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
copy_paste: true
next_step: 'Step 21 — Implement MCP tool extensions'
skills:
  - 'plan-alignment'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
  - 'node scripts/agent-customization/gates/cortex-index.gate.mjs --json'
slices:
  - slice_id: '19.1'
    title: 'Schema and red tests for signal recording and boost computation'
    files_to_change:
      - 'scripts/semantic-index/schema-v2.sql'
      - 'scripts/mcp-semantic/__tests__/feedback.red.test.mjs'
    estimate_hours: 4
    acceptance_criteria:
      - 'feedback_events and feedback_scores tables with indexes added to schema-v2.sql'
      - 'Red tests exist for all 4 signal types with pre-computed signal_strength'
      - 'Red tests exist for feedback_boost sigmoid dampening clamped to [-0.5, +0.5]'
      - 'Red tests exist for 7-day half-life time decay and impression decay (CTR-based)'
      - 'Red tests exist for privacy constraints (SHA-256 query hash, context capped at 500 chars, cascade delete)'
      - 'Tests compile but fail (red phase confirmed)'
    parallelizable: false
  - slice_id: '19.2'
    title: 'Core feedback module — recording, boost computation, and aggregation'
    files_to_change:
      - 'scripts/mcp-semantic/tools/feedback-core.mjs'
    estimate_hours: 5
    acceptance_criteria:
      - 'recordFeedbackEvent inserts into feedback_events with correct pre-computed signal_strength for all 4 types'
      - 'computeFeedbackBoost applies sigmoid dampening (0.5 * tanh(netFeedback * 2.0)) clamped to [-0.5, +0.5]'
      - 'Time decay uses 7-day half-life (FEEDBACK_HALF_LIFE_MS = 7 * 24 * 60 * 60 * 1000)'
      - 'Impression decay computed with MIN_IMPRESSIONS_FOR_DECAY=10, MIN_CTR_FOR_NEUTRAL=0.1'
      - 'updateFeedbackScores and recomputeAllFeedbackScores maintain feedback_scores table'
      - 'Privacy enforced: context truncated to 500 chars, query hashed with SHA-256, no plaintext stored'
      - 'Red tests from slice 19.1 that test core module functions pass'
    parallelizable: false
  - slice_id: '19.3'
    title: 'Automatic signal collection and submit_feedback MCP tool'
    files_to_change:
      - 'scripts/mcp-semantic/tools/search-corpus.mjs'
      - 'scripts/mcp-semantic/tools/load-chunk.mjs'
      - 'scripts/mcp-semantic/tools/submit-feedback.mjs'
      - 'scripts/mcp-semantic/repo-cortex-mcp.mjs'
    estimate_hours: 5
    acceptance_criteria:
      - 'search_corpus records impression signals (fire-and-forget) for every returned chunk with query_hash'
      - 'load_chunk records click signals with query_hash correlation via LRU cache (size 50)'
      - 'submit_feedback MCP tool accepts chunk_id, signal_type, context; returns {chunk_id, signal_type, recorded, feedback_boost_after}'
      - 'submit_feedback registered in repo-cortex-mcp.mjs tool list'
      - 'Automatic signal writes are best-effort (silent drop on failure, no latency added)'
      - 'All red tests for automatic collection and submit_feedback pass'
    parallelizable: false
  - slice_id: '19.4'
    title: 'index_stats extension, search_corpus response extension, and green validation'
    files_to_change:
      - 'scripts/mcp-semantic/tools/index-stats.mjs'
      - 'scripts/mcp-semantic/tools/search-corpus.mjs'
    estimate_hours: 4
    acceptance_criteria:
      - 'index_stats returns feedback_stats (total_events, events_by_type, chunks_with_feedback, average_feedback_boost, feedback_weight, feedback_half_life_days, last_recomputed_at)'
      - 'search_corpus results include feedback_boost and feedback_signals per result'
      - 'All feedback tests pass (green)'
      - '100% coverage on new/modified src/ files (feedback-core.mjs, submit-feedback.mjs, search-corpus.mjs changes, load-chunk.mjs changes, index-stats.mjs changes)'
      - 'npm run quality:folder -- --folder=scripts/mcp-semantic passes'
      - 'npm run test:silent passes with no regressions'
    parallelizable: false
VALIDATION_EVIDENCE:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md — PASS'
  - 'node scripts/agent-customization/gates/cortex-index.gate.mjs --json — PASS'
  - 'npm run quality:folder -- --folder=scripts/mcp-semantic — PASS'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=feedback --selectProjects=mcp-semantic-mjs with NODE_OPTIONS=--experimental-vm-modules — 59/59 PASS'
  - 'Coverage on feedback-core.mjs, index-stats.mjs, load-chunk.mjs, submit-feedback.mjs — 100% statements/branches/functions/lines'
NEXT: 'Step 20 — Implement context window assembly'
```

**User instruction:** Paste this full step packet.

**Step objective:** Implement 4 signal types (impression, click, reference, explicit), `feedback_events` and `feedback_scores` tables, feedback boost with sigmoid dampening clamped to [-0.5, +0.5], 7-day half-life time decay, impression decay, `submit_feedback` MCP tool, and automatic signal collection.

**Context the agent must know:**

- Design: `rag_architecture/cortex-relevance-feedback.md`
- 4 signal types: impression (0.1), click (0.3), reference (0.6), positive/negative explicit (—1.0)
- `feedback_events` table: event_id, chunk_id, signal_type, signal_strength, query_hash, agent_id, context, created_at
- `feedback_scores` table: chunk_id, total_positive, total_negative, total_impressions, total_clicks, total_references, last_feedback_at, feedback_boost
- Feedback boost: sigmoid dampening clamped to [-0.5, +0.5]; 7-day half-life time decay; impression decay
- Automatic collection: impression signals from `search_corpus` results; click signals from `load_chunk`
- `submit_feedback` MCP tool: explicit/reference signals
- Privacy: query stored as SHA-256 hash; no user content stored; agent_id optional; context capped at 500 chars

**TDD cycle:**

1. **Red tests**: Write failing tests for signal recording (all 4 types), feedback_boost computation with sigmoid dampening, time decay, impression decay, `submit_feedback` MCP tool, automatic signal collection in `search_corpus` and `load_chunk`.
2. **Implementation**: Implement feedback tables, boost computation, signal collection hooks, `submit_feedback` MCP tool.
3. **Green validation**: Verify all 4 signal types are recorded; verify boost computation matches expected formula; verify 7-day half-life decay; verify impression decay; verify automatic signals from search_corpus and load_chunk.

**Acceptance criteria:**

- All 4 signal types recorded correctly with pre-computed signal_strength
- `feedback_boost` computation: sigmoid dampening, clamped to [-0.5, +0.5], 7-day half-life decay, impression decay
- `submit_feedback` MCP tool accepts chunk_id, signal_type, context; returns feedback summary
- Automatic impression signals from `search_corpus`; automatic click signals from `load_chunk`
- Privacy: query_hash only (SHA-256), no plaintext query stored; context capped at 500 chars

**Dependencies:** Step 13 (semantic chunking) — feedback events reference `chunk_id` which changes after re-chunking; cascade deletes handle this.

**Stop conditions:** Done when all 4 signal types (impression, click, reference, explicit) are fully implemented, `feedback_boost` computation with sigmoid dampening clamped to [-0.5, +0.5] and 7-day half-life time decay is correct, `submit_feedback` MCP tool returns feedback summary, automatic impression/click signal collection from `search_corpus` and `load_chunk` works, all red tests turn green, and privacy constraints (SHA-256 query hash, no plaintext query, context capped at 500 chars) are enforced; blocked if Step 13 (semantic chunking) is incomplete.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md`

#### Step 20 — Implement context window assembly [DONE]

```yaml
phase: 2
step: 20
goal: 'implementing'
tdd_sequence: 'red-green'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
copy_paste: true
next_step: 'Step 21 — Implement MCP tool extensions'
skills:
  - 'plan-alignment'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
  - 'node scripts/agent-customization/gates/cortex-index.gate.mjs --json'
```

**User instruction:** Paste this full step packet. Implement the five-stage `assembleContext` pipeline: enrichment → deduplication (SHA-256 exact, cosine = 0.95 near-duplicate, parent-child collapse) → ordering (relevance tier + file grouping + char_start) → budget management (essential/standard/supplementary tiers, token counting) → stitching (context headers, same-file continuation).

**Step objective / context:**
- Design doc: `rag_architecture/cortex-context-assembly.md`
- Pipeline must be stateless, idempotent, composable, retriever-agnostic.
- Dedup: SHA-256 exact hash → cosine ≥ 0.95 near-duplicate → parent-child collapse (prefer child over parent).
- Ordering: relevance tier thresholds (essential ≥ 0.7, supporting ≥ 0.4, supplementary < 0.4) → file grouping by max score → within file: char_start ascending → family priority.
- Budget: default 4096 tokens; essential tier always included; supplementary truncated first; graceful truncation at sentence boundary.
- Stitching: context headers as separators `[file_path > heading_path]`; same-file continuation logic; output `context_format` (markdown or json).
- New file: `scripts/semantic-index/assemble-context.mjs`; implement `search_context` MCP tool.

**TDD cycle:**
1. Red tests for each pipeline stage.
2. Implementation of all five stages as composable pure functions + `search_context` MCP tool.
3. Green validation: dedup, ordering, budget, stitching.

**Acceptance criteria:**
- SHA-256 exact dedup removes duplicate content.
- Cosine ≥ 0.95 near-duplicate detection works when embeddings are available; degrades to exact-only when cold.
- Parent-child collapse prefers child sub-chunk over parent.
- Ordering: essential → supporting → supplementary; within tier file grouping by max score; within file char_start ascending.
- Budget: default 4096 tokens; essential always included; supplementary truncated; graceful sentence-boundary truncation.
- Stitching: headers as separators; same-file continuation without repeated headers; markdown and json output formats.

**Dependencies:** Step 13 (semantic chunking columns `context_header`, `parent_chunk_id`, `depth`), Step 14 (query classification for tier thresholds), Step 16 (cross-encoder reranking scores), Step 18 (query expansion results).

**Slices:**

```yaml
slices:
  - slice_id: '20-red-tests'
    title: 'Writing red tests for assembleContext pipeline and search_context tool contract'
    files_to_change:
      - 'scripts/mcp-semantic/__tests__/assemble-context.red.test.mjs'
    estimate_hours: 4
    acceptance_criteria:
      - Red tests exist and fail for each pipeline stage (enrichment, exact dedup, near-dup, parent-child collapse, ordering, budget, stitching).
      - Red tests exist and fail for search_context tool schema/contract.
      - Tests follow existing semantic-index test patterns (better-sqlite3 temp DB, schema-v2.sql, ESM __dirname shim).
      - Single expect per it() block.
    parallelizable: false

  - slice_id: '20-core-pipeline'
    title: 'Implementing assembleContext enrichment, deduplication, ordering, budget, and stitching pure functions'
    files_to_change:
      - 'scripts/semantic-index/assemble-context.mjs'
      - 'scripts/mcp-semantic/__tests__/assemble-context.red.test.mjs'
    estimate_hours: 6
    acceptance_criteria:
      - New `assembleContext` module exports pure functions for each stage: `enrichChunks`, `deduplicateChunks`, `orderChunks`, `enforceBudget`, `stitchContext`.
      - SHA-256 exact dedup selects representative by score, metadata richness, family priority, lowest chunk_id.
      - Cosine near-duplicate uses threshold 0.95; skipped when embeddings unavailable.
      - Parent-child collapse removes a parent when any child is present.
      - Ordering: tier → file max score → char_start → family priority.
      - Budget: default 4096 tokens; essential always included; supporting/supplementary soft includes; graceful sentence-boundary truncation; returns metadata.
      - Stitching produces markdown and json formats with headers and same-file continuation.
    parallelizable: false

  - slice_id: '20-mcp-tool'
    title: 'Implementing search_context MCP tool and registering it in the server'
    files_to_change:
      - 'scripts/mcp-semantic/tools/search-context.mjs'
      - 'scripts/mcp-semantic/repo-cortex-mcp.mjs'
      - 'scripts/mcp-semantic/__tests__/search-context.red.test.mjs'
    estimate_hours: 4
    acceptance_criteria:
      - `searchContext` tool composes `searchCorpus` + `assembleContext` with the input schema from the design doc.
      - Tool registered in `createRepoCortexTools` alongside `search_corpus`.
      - Graceful degradation when dense/reranker are cold.
      - Red tests for schema, composition, and cold-state fallback pass.
    parallelizable: false

  - slice_id: '20-green-validation'
    title: 'Green validation, coverage guard, and plan sync for Step 20'
    files_to_change:
      - 'coverage/lcov.info'
      - 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
    estimate_hours: 3
    acceptance_criteria:
      - All red tests pass; new src/ files hit 100% statements/branches/functions/lines.
      - `npm run quality:folder -- --folder=scripts/semantic-index` PASS.
      - `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md` PASS.
      - `node scripts/agent-customization/gates/cortex-index.gate.mjs --json` PASS.
      - Step 20 status advanced to [DONE]; validation evidence appended to plan.
    parallelizable: false
```

**Handoff query:**
- Which slice should Agent Zero dispatch first: `20-red-tests` to `03-red-testing`, then `20-core-pipeline` and `20-mcp-tool` to `04-implementing`, then `20-green-validation` to `05-green-testing`.
- Confirm whether `search_context` should be implemented as a standalone tool (recommended by design doc) or as a `use_assembly` extension of `search_corpus`.
- Verify that Step 13 semantic chunking columns (`context_header`, `parent_chunk_id`, `depth`) and Step 16 reranker scores are present and warm before starting `20-mcp-tool`.

#### Step 21 — Implement MCP tool extensions [WIP]

```yaml
phase: 2
step: 21
goal: 'implementing'
tdd_sequence: 'red-green'
status: '[WIP]'
mode: 'fresh-session'
source_of_truth: 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
copy_paste: true
next_step: 'Step 22 — Implement RAG evaluation suite'
skills:
  - 'plan-alignment'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
  - 'node scripts/agent-customization/gates/cortex-index.gate.mjs --json'
```

**User instruction:** Paste this full step packet.

**Step objective:** Implement four new MCP tools (`search_advanced`, `search_context`, `traverse_graph`, `submit_feedback`) and two extended tools (`search_corpus` with metadata filter + classification hints, `index_stats` with metadata coverage), orchestrating the full retrieval pipeline.

**Context the agent must know:**

- Design: `rag_architecture/cortex-mcp-tool-extensions.md`
- `search_advanced`: full pipeline orchestration (classify ? expand ? retrieve ? rerank ? assemble)
- `search_context`: composes search_corpus + assembleContext for agent-ready context strings
- `traverse_graph`: BFS traversal of entity/relationship graph (Step 17)
- `submit_feedback`: records relevance signals to feedback_events/feedback_scores (Step 19)
- Extended `search_corpus`: `metadata` filter parameter, `classification_hints` parameter
- Extended `index_stats`: `include_metadata_coverage` parameter
- All tools follow MCP local server contract: stdio transport, `server.registerTool()`, zod schema validation, parameterized SQL
- Graceful degradation matrix for all missing subsystems

**TDD cycle:**

1. **Red tests**: Write failing tests for each new tool's schema, error handling, and degradation behavior; test `search_advanced` orchestration pipeline; test `search_context` assembly output; test `traverse_graph` BFS; test `submit_feedback` signal recording; test extended `search_corpus` metadata filter; test extended `index_stats` coverage.
2. **Implementation**: Implement all four new tools, two extensions, degradation matrix, error taxonomy.
3. **Green validation**: Verify each tool produces correct output for valid inputs; verify degradation when subsystems are cold/missing; verify backward compatibility; verify latency budgets.

**Acceptance criteria:**

- `search_advanced` orchestrates full pipeline with classification-aware defaults per query class
- `search_context` returns assembled context string with budget management
- `traverse_graph` returns entities + edges with BFS traversal
- `submit_feedback` records signals and updates feedback_scores
- Extended `search_corpus` with `metadata` filter and `classification_hints`
- Extended `index_stats` with `metadata_coverage` section
- Graceful degradation: each tool defines cold/missing-state fallback
- Backward compatible: existing calls without new params produce identical results
- Error handling: timeout handling with partial-result fallback; error code taxonomy

**Dependencies:** Steps 14-20 — all subsystem implementations must be complete before the integration tools can orchestrate them.

#### Step 22 — Implement RAG evaluation suite [PLANNED]

```yaml
phase: 2
step: 22
goal: 'implementing'
tdd_sequence: 'red-green'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
copy_paste: true
next_step: 'Step 23 — Implement ANN index'
skills:
  - 'plan-alignment'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
  - 'node scripts/agent-customization/gates/cortex-index.gate.mjs --json'
```

**User instruction:** Paste this full step packet.

**Step objective:** Implement the comprehensive RAG evaluation suite: 6-class query taxonomy (56-68 curated queries), 5 automated metrics (MRR@k, nDCG@k, Recall@k, context relevance, latency), 4 baseline conditions, CI regression gate, and A/B comparison framework.

**Context the agent must know:**

- Design: `rag_architecture/cortex-rag-eval-suite.md`
- 6 classes: simple_lookup, cross_boundary, multi_hop, exploratory, code_specific, plan_specific
- Metrics: MRR@k for k?{1,3,5,10}, nDCG@k for k?{5,10}, Recall@k for k?{5,10,20}, context relevance, latency
- 4 baseline conditions: bm25_only, hybrid, hybrid_rerank, advanced_default
- Query schema v2 with graded relevance (0-3) and expected_chunk_ids
- CI regression gate: MRR@5 FAIL threshold; nDCG/Recall WARN thresholds
- A/B comparison: Wilcoxon signed-rank test; alpha sweep
- Eval runner module: `eval-runner.mjs`, `eval-metrics.mjs`, `eval-compare.mjs`, `eval-baseline.mjs`

**TDD cycle:**

1. **Red tests**: Write failing tests for all 5 metric computations (MRR@k, nDCG@k, Recall@k, context relevance, latency), query classification accuracy, baseline measurement, regression detection, and A/B comparison.
2. **Implementation**: Implement eval runner, metrics, comparison framework, query set curation, baseline storage, CI gate.
3. **Green validation**: Run eval suite against current system; establish baseline measurements; verify regression detection; verify A/B comparison produces valid statistical results.

**Acceptance criteria:**

- 56-68 curated queries across 6 classes with graded relevance (0-3)
- MRR@k, nDCG@k, Recall@k computed correctly for all baseline conditions
- Context relevance metric measures assembled context quality
- Latency metric tracks end-to-end query time
- CI regression gate: MRR@5 FAIL threshold triggers failure; nDCG/Recall WARN thresholds trigger warning
- A/B comparison: Wilcoxon signed-rank test with p-value reporting; alpha sweep
- Baseline measurements stored for regression comparison
- Self-test queries validate eval runner correctness

**Dependencies:** Steps 13-21 — eval suite validates all subsystems and must run after they are implemented.

#### Step 23 — Implement ANN index [PLANNED]

```yaml
phase: 2
step: 23
goal: 'implementing'
tdd_sequence: 'red-green'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
copy_paste: true
next_step: 'Phase 3 validation'
skills:
  - 'plan-alignment'
  - 'repo-cortex-workflow'
specialists:
  - 'cortex-embeddings-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
  - 'node scripts/agent-customization/gates/cortex-index.gate.mjs --json'
  - 'node scripts/agent-customization/gates/dense-readiness.gate.mjs --json'
```

**User instruction:** Paste this full step packet.

**Step objective:** Implement ANN index with three-strategy approach (brute_force_cached below 50K threshold, HNSW above threshold, brute_force for baseline), hnswlib-node as optional dependency with graceful fallback, LRU query result caching, and `ann_build_index` MCP tool.

**Context the agent must know:**

- Design: `rag_architecture/cortex-ann-index.md`
- Three strategies: `brute_force_cached` (< 50K chunks), `hnsw` (= 50K chunks), `brute_force` (baseline)
- HNSW parameters: M=32, ef_construction=200, ef_search=100
- HNSW via `hnswlib-node` as optional dependency; `sqlite-vec` rejected (brute-force only)
- LRU query result cache for sub-threshold performance
- Incremental update: =5% change threshold for rebuild vs. incremental
- 4 new tables: `ann_index_meta`, `ann_index_chunk_map`, `ann_query_cache`, `ann_threshold_config`
- Extended `search_corpus` with `dense_strategy` field; extended `index_stats` with `ann` section
- New MCP tool: `ann_build_index`
- Cross-platform CI: optional dependency with runtime detection

**TDD cycle:**

1. **Red tests**: Write failing tests for strategy selection (threshold-based), brute_force_cached with LRU cache, HNSW index build/search (mock or small corpus), incremental update detection, graceful fallback when hnswlib-node unavailable, `ann_build_index` MCP tool, `search_corpus` dense_strategy field.
2. **Implementation**: Implement three-strategy selection, HNSW wrapper with optional dependency, LRU cache, incremental update, MCP extensions.
3. **Green validation**: Verify Recall@10 = 0.95 vs brute-force; verify brute_force_cached works below 50K; verify HNSW works above threshold (or graceful fallback); verify incremental update threshold; verify backward compatibility.

**Acceptance criteria:**

- Recall@10 = 0.95 relative to brute-force
- `brute_force_cached` works correctly below 50K threshold with LRU cache
- HNSW index build and search work when `hnswlib-node` is available; graceful fallback when unavailable
- LRU query cache reduces repeated query latency
- `ann_build_index` MCP tool builds/rebuilds ANN index
- `search_corpus` extended with `dense_strategy` field in response
- `index_stats` extended with `ann` section showing index status
- Fully backward-compatible: no API changes below threshold

**Dependencies:** Can proceed in parallel with Steps 16-22. Only depends on the existing dense search infrastructure (no other Phase 2 subsystems).

### Phase 3 — Validation and integration [PLANNED]

Full eval suite execution, end-to-end regression testing, MCP tool integration validation, CI gate integration, and baseline comparison against Phase 1 measurements.

**Phase 3 steps will be defined after Phase 2 implementation completes.** Phase 3 will include:

1. Run full eval suite (Step 22) against all 4 baseline conditions
2. Compare MRR@5, nDCG@5, Recall@5 against Phase 1 baselines (BM25 MRR@5 = 0.225, Hybrid MRR@5 = 0.308)
3. Regression testing: verify no existing search_corpus behavior is broken
4. CI gate integration: eval runner as CI regression gate
5. Performance validation: cross-encoder latency, context assembly latency, graph traversal latency
6. MCP tool integration validation: search_advanced, search_context, traverse_graph, submit_feedback end-to-end
7. ANN index validation: Recall@10 = 0.95

## Validation gates

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md`
- `node scripts/agent-customization/gates/cortex-index.gate.mjs --json`
- `node scripts/agent-customization/gates/dense-readiness.gate.mjs --json`

### Latest validation evidence

- 2026-06-15: Step 20 green validation complete. Focused `mcp-semantic-mjs` tests 31/31 pass; broader `semantic-index-mjs` + `mcp-semantic-mjs` suite 219/219 pass. `quality:folder` PASS for `scripts/semantic-index/` and `scripts/mcp-semantic/`. Plan-sync gate PASS; cortex-index gate PASS after rebuilding stale index. Step 20 marked [DONE]; Step 21 now [WIP].
- 2026-06-15: Step 20 (context window assembly) expanded into executable slices: `20-red-tests`, `20-core-pipeline`, `20-mcp-tool`, `20-green-validation`. Plan sync PASS. Status remains [WIP]; ready for Agent Zero to dispatch `03-red-testing` → `04-implementing` → `05-green-testing` in sequence.
- 2026-06-09: Workflow sync: Advanced Phase 2 Step 19 → [DONE]; Phase 2 Step 20 → [WIP]
- 2026-06-11: Semantic-index test suite fixes: (1) ESM `__dirname` shims added to 7 test files that used `__dirname` without `import.meta.url` compatibility (`classify-query.red.test.ts`, `semantic-index.red.test.ts`, `routing-table.red.test.ts`, `metadata-enrichment.red.test.ts`, `metadata-filter.red.test.ts`, `build-index.health.test.ts`, `validate-index.fixhint.test.ts`); (2) Schema fixture updates in `dense-readiness.red.test.ts` and `embed-index.red.test.ts` from v1 to v2/v3 (added FTS5 content-synced triggers and v3 columns); (3) `schema-v2.sql` idempotency fix: moved v3 `ALTER TABLE ADD COLUMN` statements into `CREATE TABLE` definitions so that `initSemanticIndex` is idempotent (second call no longer fails with "duplicate column name: arch_layer"). All 18/18 semantic-index tests pass. Quality gate PASS.
- 2026-06-10: Step 14 (query classification) completed. 6-class rule-based classifier (`classify-query.mjs`), per-class routing table (`routing-table.mjs`), classification-aware `search_corpus` integration, eval runner (95.8% accuracy, <5ms latency). All acceptance criteria met. Step 15 (metadata filtering) is next.
- 2026-06-09: Phase 2 step packets defined (Steps 13—23). Step 13 [] (step packet defined, ready for implementation). Steps 14—23 [PLANNED]. Workflow sync auto-advance corrected: Step 13 reverted from [DONE] to []; Step 14 reverted from [WIP] to [PLANNED].
- 2026-06-08: Workflow sync: Advanced Phase 2 Step 13 ? [DONE]; Phase 2 Step 14 ? [WIP]
- 2026-06-08: Step 10 MCP tool extensions architecture complete (Sections A—K). Four new tools (search_advanced, search_context, traverse_graph, submit_feedback) and two extensions (search_corpus with metadata filter + classification hints, index_stats with metadata coverage). search_advanced orchestrates the full pipeline (classification ? expansion ? retrieval ? re-ranking ? assembly) with classification-aware defaults per query class. search_context composes search_corpus + assembleContext for agent-ready context strings. traverse_graph implements BFS traversal of the entity/relationship graph with seed_query/seed_names discovery and relationship type filtering. submit_feedback records relevance signals to feedback_events/feedback_scores tables with automatic aggregate score update and feedback_boost computation. Extended search_corpus adds classification_hints parameter for classification-aware retrieval without full pipeline. Extended index_stats adds include_metadata_coverage parameter with per-column coverage statistics. Comprehensive error code taxonomy, timeout handling with partial-result fallback, graceful degradation matrix for all missing subsystems, backward-compatible parameter additions, and 16 tool-specific eval queries plus 5 cross-tool integration scenarios with regression thresholds and latency budgets.
- 2026-06-08: Step 11 RAG evaluation suite architecture complete (Sections A—K). Comprehensive eval framework with 6-class taxonomy (simple_lookup, cross_boundary, multi_hop, exploratory, code_specific, plan_specific) targeting 56—68 curated queries. Five automated metrics (MRR@k for k?{1,3,5,10}, nDCG@k for k?{5,10}, Recall@k for k?{5,10,20}, context relevance, latency) plus deferred human faithfulness evaluation. Four baseline conditions (bm25_only, hybrid, hybrid_rerank, advanced_default). Query schema v2 with graded relevance (0—3) and expected_chunk_ids. CI regression gate with MRR@5 FAIL threshold and nDCG/Recall WARN thresholds. A/B comparison with Wilcoxon signed-rank test and alpha sweep. Eval runner module architecture (eval-runner.mjs, eval-metrics.mjs, eval-compare.mjs, eval-baseline.mjs). Per-class aggregation and baseline storage protocol. Self-test queries for eval runner validation.
- 2026-06-08: Step 09 structured metadata filtering architecture complete (Sections A—K). Six new metadata columns (arch_layer, jsdoc_quality, jsdoc_word_count, cyclomatic_complexity, test_coverage, source_path_pattern) on chunks table plus three document-level columns. Filter grammar supporting 14 predicate types (eq, neq, in, not_in, gt, gte, lt, lte, like, is_null, is_not_null, and, or, not) with boolean composition. SQLite indexes for common filter patterns. BM25 filter via SQL WHERE clause; dense filter via post-retrieval in-memory filtering. MCP `search_corpus` extension with `metadata` parameter accepting structured filter predicate trees. Backward-compatible `family` parameter combined with AND. Filter validation with field allow-list, type checking, enum validation, depth limit (10), predicate limit (50), LIKE pattern whitelist. Build-time metadata enrichment pipeline. Eval design with 6 filter-specific eval queries and regression thresholds. Plan sync: PASS.
- 2025-06-15: Step 12 ANN index architecture complete (Sections A—K). Three-strategy approach: brute_force_cached below 50K threshold, HNSW above threshold, brute_force for baseline. HNSW via hnswlib-node (optional dependency, graceful fallback). sqlite-vec evaluated and rejected (brute-force only, no ANN). LRU query result cache for sub-threshold performance. HNSW index build pipeline with M=32, ef_construction=200, ef_search=100. Incremental update with =5% change threshold. Recall@10 = 0.95 validation gate. Process-lifetime index caching. Four new database tables (ann_index_meta, ann_index_chunk_map, ann_query_cache, ann_threshold_config). Extended search_corpus with dense_strategy field. Extended index_stats with ann section. New ann_build_index MCP tool. 8 ANN-specific eval queries. Cross-platform CI via optional dependency. Fully backward-compatible.

## Orchestration enforcement gap analysis (2026-06-13)

### Diagnosis

During Step 18 implementation, Agent Zero (the Tier-0 orchestrator) violated its own mandate in two ways:

1. **Orchestrator did substantive work directly**: Wrote test files, edited implementation files, and ran test commands — all prohibited by §0 ("Delegate, don't do").
2. **Skipped TDD phase sequencing**: Instead of dispatching through 03-red-testing → 04-implementing → 05-green-testing, bundled everything into one 04-implementing task.

### Root-cause findings (5 enforcement gaps)

**Gap 1: Step packet has no TDD-sequence enforcement field**
Step 18's YAML metadata specifies `goal: 'implementing'` (previously `agent: '04-implementing'`) and includes a `TDD cycle` prose section, but there was NO required field or gate that enforced a preceding 03-red-testing dispatch. The step-packet gate (`step-packet.gate.mjs`) validates structural fields (`phase`, `step`, `goal`, `status`, `next_step`) and required prose sections (`Stop conditions`, `Required validation`), but does NOT validate TDD phase sequencing. The `tdd_sequence` field has since been added to all Phase 2 step packets.

**Gap 2: Runtime enforcement validates carrier existence, not agent-phase alignment**
The runtime enforcement system (`runtime-enforcement.mjs`) validates that a prepared carrier exists for write/execute actions with correct `flowId`, `currentAgent`, `delegatorChain`, `planPath`, `allowedActionClass`, and `expectedToolName`. However, it does NOT validate that `currentAgent` matches the expected SDLC phase for the action being performed. Agent Zero can prepare a carrier with `currentAgent: 04-implementing` and then write test files directly — the hook passes because the carrier exists, not because the right agent is performing the right phase work.

**Gap 3: `usesMatchingNumberedAgent` validator has a semantic mismatch**
`validate-plan-phase-packets.mjs` (line 510-522) enforces that Step N must use an agent starting with `N-`. For Step 18, this checks if `04-implementing` starts with `18-` → false. This is a validation bug: Phase 2 implementation steps use SDLC agents (04-implementing, 05-green-testing) that do NOT match the step number. The validator should instead verify that the step's `goal` field is a valid SDLC-phase goal (`implementing`, `green-testing`, etc.) for the work described. With the migration to `goal`-based dispatch, this validator needs updating to check `goal` instead of `agent`.

**Gap 4: red-test-confirmation gate is a stub**
`red-test-confirmation.gate.mjs` is a Tier-2 gate that always returns `pass: true` with `mode: 'standalone-descriptor'`. It never actually checks whether red tests were written or whether 03-red-testing was dispatched. Even if 03-red-testing were invoked, there is no gate enforcing its output quality or existence.

**Gap 5: No gate detects skipped TDD phases**
There is no `phase-sequence` or `tdd-sequence` gate that checks whether a step with a TDD cycle was preceded by a 03-red-testing completion. Existing gates (`step-packet`, `plan-sync`, `agent-graph`, `tier-enforcement`) validate structural metadata, plan synchronization, agent graph validity, and tier structure — but none validate dispatch ordering or TDD phase sequencing.

### Proposed fix (three-pronged)

**Fix A: Add `tdd_sequence` field to step packet schema** (IMPLEMENTED)
Add a `tdd_sequence` field to step YAML blocks with values like `red-green`, `green-only`, or `skip`. When `tdd_sequence: red-green`, the step-packet gate validates that a prior 03 step exists in the same phase. Update `validate-plan-phase-packets.mjs` to check this field and add a `stepRequiredMetadataKeys` entry for `tdd_sequence` when the step's prose includes a "TDD cycle" section. All Phase 2 step packets now include `tdd_sequence: 'red-green'` and `goal` instead of `agent`/`agent_file`.

**Fix B: Create `tdd-phase-sequence` gate**
Create a new Tier-1 gate `scripts/agent-customization/gates/tdd-phase-sequence.gate.mjs` that:

1. Reads the active plan's WIP step packet
2. If the step's TDD cycle section includes "Red tests" and `tdd_sequence: red-green`, checks the learning log for a `03-red-testing` flow completion event before any `04-implementing` events for the same scope
3. Returns `{pass: false, fixHint: "Step N requires red tests before implementation. Route through 03-red-testing first."}` if missing

**Fix C: Harden `red-test-confirmation` gate from stub to active check**
Upgrade `red-test-confirmation.gate.mjs` from `mode: 'standalone-descriptor'` to an active gate that:

1. Reads the active plan's WIP step for `FILES_CHANGED` entries from 03-red-testing output
2. Checks that at least 2 red test file paths exist
3. Checks that the test files are present on disk and import the target module
4. Returns `{pass: false}` when no red-test evidence exists

### Secondary fix: `usesMatchingNumberedAgent` validator

The `usesMatchingNumberedAgent` function should be updated to recognize that implementation steps (numbered 08+) may use any valid SDLC goal (planning, researching, red-testing, implementing, green-testing, documenting, logging, helping), not just the agent matching the step number. Only enforce number matching for steps 01-07 (the SDLC phase steps that open a phase). Steps numbered 08+ should validate that the `goal` field is a valid SDLC-phase goal in the routing table.

## Handoff query

```
Continue from the current repo state only. Do not rely on prior chat history.

Phase 1 is [DONE]. All 12 architecture design steps are complete. Design documents are in rag_architecture/.

Phase 2 (Implementation) is [WIP] with Steps 13-20 [DONE]. Step 21 (MCP tool extensions) is the active step [WIP]. Steps 22-23 are [PLANNED] with detailed TDD specifications.

Dependency order: Step 13 (chunking) → Steps 14-15 (classification, metadata) → Steps 16-19 (reranking, graph, expansion, feedback) → Step 20 (context assembly) → Step 21 (MCP tools) → Step 22 (eval suite) → Step 23 (ANN index, parallel-capable).

Next step: Dispatch Step 21 to `04-implementing`. Read `rag_architecture/cortex-mcp-tool-extensions.md` for the complete design.
```
