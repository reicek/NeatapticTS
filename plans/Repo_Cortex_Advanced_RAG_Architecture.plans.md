# Repo Cortex Advanced RAG Architecture

**Status:** [WIP]

## Scope

Design and implement a no-compromise advanced RAG system for the NeatapticTS Repo Cortex. The current Cortex (Layers 1–6) provides a solid foundation: SQLite-backed BM25 full-text search, ONNX local dense embeddings (`all-MiniLM-L6-v2`), hybrid BM25+dense ranking, freshness proofs, and MCP tool exposure. However, the current system lacks the advanced retrieval, re-ranking, context assembly, and query-understanding capabilities required for high-quality agent-assisted development. This plan audits the existing system, identifies every gap, and designs the architecture to close them with no compromises.

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
    criterion: 'Complete gap analysis of current Cortex Layers 1–6 against advanced RAG requirements.'
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

| Gap | Severity | Architecture design |
|---|---|---|
| Chunking quality | **CRITICAL** | [Semantic chunking](../rag_architecture/cortex-semantic-chunking.md) |
| Embedding model | Small model, truncation | Deferred (chunking quality is #1) |
| Ranking pipeline | Fixed alpha, no adaptation | [Query classification](../rag_architecture/cortex-query-classification.md) + [Cross-encoder](../rag_architecture/cortex-cross-encoder-reranking.md) |
| Context assembly | None | [Context assembly](../rag_architecture/cortex-context-assembly.md) |
| Metadata enrichment | Only family filter | [Metadata filtering](../rag_architecture/cortex-metadata-filtering.md) |
| Multi-hop | Single-pass only | [Entity graph](../rag_architecture/cortex-entity-graph.md) |
| Cross-encoder | Bi-encoder only | [Cross-encoder](../rag_architecture/cortex-cross-encoder-reranking.md) |
| Query expansion | No synonym discovery | [Query expansion](../rag_architecture/cortex-query-expansion.md) |
| Relevance feedback | No feedback mechanism | [Relevance feedback](../rag_architecture/cortex-relevance-feedback.md) |
| Structured filtering | Only family filter | [Metadata filtering](../rag_architecture/cortex-metadata-filtering.md) |
| MCP tool extensions | 7 basic tools | [MCP tools](../rag_architecture/cortex-mcp-tool-extensions.md) |
| RAG eval suite | 20 queries, no systematic eval | [RAG eval](../rag_architecture/cortex-rag-eval-suite.md) |
| ANN index | Deferred | Step 12 (planned) |

**Baseline measurements:** BM25 MRR@5 = 0.225, Hybrid MRR@5 = 0.308, improvement = +0.083

---

## Implementation phases

### Phase 1 — Architecture investigation and design [DONE]

#### Step 01 — Audit current Cortex against advanced RAG requirements [DONE]

> **Full audit investigation and findings:** [rag_architecture/cortex-current-system-audit.md](../rag_architecture/cortex-current-system-audit.md)

**Audit completion summary:**
- ✅ 1. Chunking quality: CRITICAL — ts-source chunks up to 42K chars, 51.4% lack heading_path, 2,604 empty stubs
- ✅ 2. Embedding model: NOT primary bottleneck — chunking quality is #1 issue
- ✅ 3. Ranking pipeline: Fixed alpha=0.5 suboptimal; query-length heuristic + cross-encoder recommended
- ✅ 4. Context assembly: None — designed assemble_context pipeline
- ✅ 5. Metadata enrichment: 6 key fields available but not indexed
- ✅ 6. Multi-hop: 3-hop iterative retrieval with diminishing-relevance stopping
- ✅ 7. Cross-encoder: ms-marco-MiniLM-L-6-v2 recommended (~22MB, ~5ms/pair)
- ✅ 8. ANN index: DEFERRED — brute-force acceptable at 31K scale
- ✅ 9. RAG eval suite: 50+ query taxonomy with MRR@5, nDCG@5, Recall@5 metrics

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

**Design summary:** ANN index architecture with three-strategy approach (brute_force_cached below 50K threshold, HNSW above threshold, brute_force for baseline). HNSW via hnswlib-node as optional native dependency with graceful fallback. sqlite-vec evaluated and rejected (brute-force only, no ANN acceleration). Query result LRU caching for sub-threshold performance. Incremental update strategy with staleness detection and ≤5% change incremental path. Recall@10 ≥ 0.95 validation gate. Process-lifetime index caching. Extended search_corpus response with dense_strategy field. Extended index_stats with ann section. New ann_build_index MCP tool. Four new database tables (ann_index_meta, ann_index_chunk_map, ann_query_cache, ann_threshold_config). Cross-platform CI via optional dependency with runtime detection. Fully backward-compatible — no API changes below threshold.

### Phase 2 — Implementation (designs from Phase 1) [PLANNED]

Implementation steps will be defined after Phase 1 design completes. Each Phase 1 step that approves an architecture will produce a corresponding Phase 2 implementation step following the TDD red → green → coverage cycle.

#### Step 13 — Implement semantic chunking [PLANNED]

```yaml
phase: 2
step: 13
agent: '04-implementing'
status: '[PLANNED]'
skills: 'plan-alignment'
```

#### Step 14 — Implement query classification [PLANNED]

```yaml
phase: 2
step: 14
agent: '04-implementing'
status: '[PLANNED]'
skills: 'plan-alignment'
```

### Phase 3 — Validation and integration [PLANNED]

Full eval suite execution, regression testing, MCP tool validation, and CI gate integration.

## Validation gates

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md`
- `node scripts/agent-customization/gates/cortex-index.gate.mjs --json`
- `node scripts/agent-customization/gates/dense-readiness.gate.mjs --json`


### Latest validation evidence

- 2026-06-08: Workflow sync: Advanced Phase 2 Step 13 → [DONE]; Phase 2 Step 14 → [WIP]
- 2026-06-08: Step 10 MCP tool extensions architecture complete (Sections A–K). Four new tools (search_advanced, search_context, traverse_graph, submit_feedback) and two extensions (search_corpus with metadata filter + classification hints, index_stats with metadata coverage). search_advanced orchestrates the full pipeline (classification → expansion → retrieval → re-ranking → assembly) with classification-aware defaults per query class. search_context composes search_corpus + assembleContext for agent-ready context strings. traverse_graph implements BFS traversal of the entity/relationship graph with seed_query/seed_names discovery and relationship type filtering. submit_feedback records relevance signals to feedback_events/feedback_scores tables with automatic aggregate score update and feedback_boost computation. Extended search_corpus adds classification_hints parameter for classification-aware retrieval without full pipeline. Extended index_stats adds include_metadata_coverage parameter with per-column coverage statistics. Comprehensive error code taxonomy, timeout handling with partial-result fallback, graceful degradation matrix for all missing subsystems, backward-compatible parameter additions, and 16 tool-specific eval queries plus 5 cross-tool integration scenarios with regression thresholds and latency budgets.
- 2026-06-08: Step 11 RAG evaluation suite architecture complete (Sections A–K). Comprehensive eval framework with 6-class taxonomy (simple_lookup, cross_boundary, multi_hop, exploratory, code_specific, plan_specific) targeting 56–68 curated queries. Five automated metrics (MRR@k for k∈{1,3,5,10}, nDCG@k for k∈{5,10}, Recall@k for k∈{5,10,20}, context relevance, latency) plus deferred human faithfulness evaluation. Four baseline conditions (bm25_only, hybrid, hybrid_rerank, advanced_default). Query schema v2 with graded relevance (0–3) and expected_chunk_ids. CI regression gate with MRR@5 FAIL threshold and nDCG/Recall WARN thresholds. A/B comparison with Wilcoxon signed-rank test and alpha sweep. Eval runner module architecture (eval-runner.mjs, eval-metrics.mjs, eval-compare.mjs, eval-baseline.mjs). Per-class aggregation and baseline storage protocol. Self-test queries for eval runner validation.
- 2026-06-08: Step 09 structured metadata filtering architecture complete (Sections A–K). Six new metadata columns (arch_layer, jsdoc_quality, jsdoc_word_count, cyclomatic_complexity, test_coverage, source_path_pattern) on chunks table plus three document-level columns. Filter grammar supporting 14 predicate types (eq, neq, in, not_in, gt, gte, lt, lte, like, is_null, is_not_null, and, or, not) with boolean composition. SQLite indexes for common filter patterns. BM25 filter via SQL WHERE clause; dense filter via post-retrieval in-memory filtering. MCP `search_corpus` extension with `metadata` parameter accepting structured filter predicate trees. Backward-compatible `family` parameter combined with AND. Filter validation with field allow-list, type checking, enum validation, depth limit (10), predicate limit (50), LIKE pattern whitelist. Build-time metadata enrichment pipeline. Eval design with 6 filter-specific eval queries and regression thresholds. Plan sync: PASS.
- 2025-06-15: Step 12 ANN index architecture complete (Sections A–K). Three-strategy approach: brute_force_cached below 50K threshold, HNSW above threshold, brute_force for baseline. HNSW via hnswlib-node (optional dependency, graceful fallback). sqlite-vec evaluated and rejected (brute-force only, no ANN). LRU query result cache for sub-threshold performance. HNSW index build pipeline with M=32, ef_construction=200, ef_search=100. Incremental update with ≤5% change threshold. Recall@10 ≥ 0.95 validation gate. Process-lifetime index caching. Four new database tables (ann_index_meta, ann_index_chunk_map, ann_query_cache, ann_threshold_config). Extended search_corpus with dense_strategy field. Extended index_stats with ann section. New ann_build_index MCP tool. 8 ANN-specific eval queries. Cross-platform CI via optional dependency. Fully backward-compatible.

## Handoff query

```
Continue from the current repo state only. Do not rely on prior chat history.

Phase 1 is [DONE]. All 12 architecture design steps are complete. Design documents are in rag_architecture/.

Phase 2 (Implementation) is [PLANNED] and needs step packets defined. The Phase 1 designs that require implementation are:
- Step 02: Semantic chunking → rag_architecture/cortex-semantic-chunking.md
- Step 03: Query classification → rag_architecture/cortex-query-classification.md
- Step 04: Cross-encoder re-ranking → rag_architecture/cortex-cross-encoder-reranking.md
- Step 05: Context window assembly → rag_architecture/cortex-context-assembly.md
- Step 06: Entity/relationship graph → rag_architecture/cortex-entity-graph.md
- Step 07: Query expansion → rag_architecture/cortex-query-expansion.md
- Step 08: Relevance feedback → rag_architecture/cortex-relevance-feedback.md
- Step 09: Structured metadata filtering → rag_architecture/cortex-metadata-filtering.md
- Step 10: MCP tool extensions → rag_architecture/cortex-mcp-tool-extensions.md
- Step 11: RAG evaluation suite → rag_architecture/cortex-rag-eval-suite.md
- Step 12: ANN index → rag_architecture/cortex-ann-index.md

Next step: Define Phase 2 implementation steps with TDD red → green → coverage cycle for each design.
```
