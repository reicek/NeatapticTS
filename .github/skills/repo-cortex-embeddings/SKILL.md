---
name: repo-cortex-embeddings
description: 'Use when mapping ONNX embedding model cache state, embeddings index readiness, hybrid BM25+dense ranking gaps, or deciding whether an embeddings issue belongs to repo-cortex-embeddings. Hands off to the embeddings skill when advanced RAG is ready. Keywords: cortex embeddings, ONNX embeddings, dense retrieval, embed-index, sqlite-vec, hybrid ranking, MRR, embeddings scout.'
argument-hint: 'Describe the embeddings surface: cache state, index readiness, ranking gap, or architecture question, and whether the issue is cache, index, or architecture.'
user-invocable: false
disable-model-invocation: false
---

# Repo Cortex Embeddings

Use this skill for Repo Cortex embeddings readiness assessment only: ONNX model
cache state, embeddings index readiness, hybrid BM25+dense ranking gaps, and
deciding whether an embeddings issue belongs to `repo-cortex-embeddings` or to
a future advanced RAG layer. It does not own corpus-changing source edits, model
downloads, or index rebuilds — use it for read-only reconnaissance and handoff
preparation.

## When to Use

- The ONNX embedding model cache is missing, stale, or returns unexpected
  dimensions or errors during `npm run index:prewarm` or
  `search_corpus` dense queries.
- `neataptic-cortex-mcp:index_stats` shows `dense_state: cold` or
  `dense_state: model-only` and you need to classify the gap before taking
  action.
- Hybrid BM25+dense ranking returns results that are obviously worse than
  BM25-only for known-good queries, suggesting an alpha-blend or embedding
  quality issue.
- `neataptic-cortex-mcp:search_corpus` with `use_dense: true` returns
  `dense_degraded: true` or `dense_reason` indicating a dense search fallback.
- You need to decide whether an embeddings issue belongs to the current
  BM25+dense baseline (`repo-cortex-embeddings`) or to the future advanced RAG
  system (Layer 7+).

## Task Packet

Pass a compact packet with the observed surface, known scope, and the
assessment you need at the end.

```text
Use repo-cortex-embeddings for dense-search readiness assessment.
Observed surface: search_corpus returns dense_degraded=true with reason "model not loaded".
Known scope: prewarm runs without errors but dense_state stays model-only.
Desired assessment: classify whether this is a cache, index, or architecture gap, and prepare handoff evidence.
```

## Required Workflow

1. Check current embeddings index state:
   `neataptic-cortex-mcp:index_stats`
   Read the `dense_state`, `family_counts`, and `last_indexed` fields.
   Classify the current readiness: `cold`, `model-only`, `warm`, or `hot`.

2. Run a freshness check for the relevant files:
   `neataptic-cortex-mcp:freshness_check`
   Pass the file path for the ONNX model cache, embeddings store, or corpus
   index depending on which surface is in question.

3. Identify the ranking gap, if any:
   `neataptic-cortex-mcp:search_corpus`
   Run a known-good query with `use_dense: true` and `use_dense: false`.
   Compare result quality, MRR signals, and any `dense_degraded` or
   `dense_reason` fields in the response.

4. Classify the issue into exactly one category:
   - **Cache**: ONNX model file missing, stale, or wrong dimensions.
   - **Index**: Embeddings store empty, partially built, or schema-mismatched.
   - **Architecture**: Alpha-blend misconfigured, ranking pipeline broken, or
     advanced RAG capability (cross-encoder, query expansion, semantic chunking)
     needed beyond the current BM25+dense baseline.

5. Prepare a compact handoff paragraph naming:
   - The active surface and its current state.
   - The observed blocker or gap.
   - The smallest useful next pass (cache rebuild, index rebuild, architecture
     escalation to Layer 7+, or no action needed).

## Guardrails

- Stay read-only. Do not create caches, rebuild indices, or edit files.
- Treat the current BM25+dense hybrid as the baseline. Do not propose advanced
  RAG changes (cross-encoder re-ranking, semantic chunking, query expansion)
  through this skill — route those to the Layer 7+ plan at
  `plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md`.
- Do not assume the advanced RAG system is implemented. The current system is
  BM25 + all-MiniLM-L6-v2 dense embeddings with alpha-blend ranking.
- When `dense_state` is `cold` or `model-only`, the correct action is
  `npm run index:prewarm`. If prewarm fails, classify as a cache issue and
  prepare handoff — do not attempt manual model downloads or SQLite surgery.
- Do not conflate embeddings readiness with general Cortex index freshness.
  If the BM25 index itself is stale, route to `repo-cortex-workflow` instead.

## Expected Final Output

A strong embeddings assessment should report:

- **Embeddings surface:** one short line naming the active boundary (cache,
  index, or architecture).
- **Current state:** `dense_state` value from `index_stats`, with
  `family_counts` and `last_indexed` if relevant.
- **Readiness signals:** 2–4 short bullets confirming what is working or
  degraded.
- **Observed blockers:** 0–4 short bullets naming specific gaps.
- **Classification:** cache, index, or architecture.
- **Handoff paragraph:** one short paragraph with the active target, the
  blocker, and the smallest focused next pass.