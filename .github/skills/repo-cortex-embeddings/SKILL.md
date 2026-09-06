---
name: repo-cortex-embeddings
description: 'Use when: mapping Cortex embeddings index state, freshness, or hybrid search.'
argument-hint: 'Describe the embeddings surface: cache state, index readiness, ranking gap, or architecture question, and whether the issue is cache, index, or architecture.'
user-invocable: false
disable-model-invocation: false
skills:
  - repo-cortex-workflow
  - research-methodology
  - mcp-local-server-workflow
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

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
  BM25-only for known-good queries, suggesting an RRF fusion or embedding
  quality issue.
- `neataptic-cortex-mcp:search_corpus` with `use_dense: true` returns
  `dense_degraded: true` or `dense_reason` indicating a dense search fallback.
- You need to decide whether an embeddings issue belongs to the current
  BM25+dense baseline (`repo-cortex-embeddings`) or to the future advanced RAG
  system (Layer 7+).

## When NOT to use

Do NOT use for general corpus search or indexing - use `repo-cortex-workflow` instead. Do NOT use for research workflows - use `research-methodology` instead.

## Workflow Diagram

```text
Flowchart summary: "Embedding request" → "Check ONNX model cache"; "Check ONNX model cache" → "Model loaded?"; "Model loaded?" → "Generate embeddings" (Yes), "Load model" (No); "Generate embeddings" → "Update Turso vector index (DiskANN)"; "Load model" → "Generate embeddings"; "Update Turso vector index (DiskANN)" → "Index warm?"; "Index warm?" → "Hybrid search available" (Yes), "BM25 only fallback" (No); "Hybrid search available"; "BM25 only fallback".
```

## Degraded-state response

When `dense_state` is `cold` or `model-only`, or when a Cortex MCP tool returns
a `self_heal` block, treat that block as the authoritative signal. The
`self_heal` response contract is documented in the `repo-cortex-workflow`
skill under **Self-heal surface**. Use it to decide whether to wait
(`started`, `in_flight`, `cooldown`), run the supplied `manual_recovery`
commands (`exhausted`, `disabled`), or simply run `npm run index:prewarm`
when no self-heal block is present. Do not invent parallel repair sequences
or perform manual Turso surgery.

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
   - **Architecture**: RRF fusion misconfigured, ranking pipeline broken, or
     advanced RAG capability (cross-encoder, query expansion, semantic chunking)
     needed beyond the current BM25+dense baseline.

5. Prepare a compact handoff paragraph naming:
   - The active surface and its current state.
   - The observed blocker or gap.
   - The smallest useful next pass (cache rebuild, index rebuild, architecture
     escalation to Layer 7+, or no action needed).

## Decision Tree: Cache, Index, or Architecture

```text
Flowchart summary: "Embeddings issue" → "What kind?"; "What kind?" → "Check ONNX cache" (Model not loading), "Rebuild DiskANN vector index" (Index not ready), "Check BM25 + dense blend" (Hybrid ranking gap), "Validate recall vs brute-force fallback" (MRR regression); "Check ONNX cache"; "Rebuild DiskANN vector index"; "Check BM25 + dense blend"; "Validate recall vs brute-force fallback".
```

## Before / After Examples

**Before:**

```text
search_corpus returned dense_degraded=true.
```

**After:**

```text
search_corpus returned dense_degraded=true (reason: "model not loaded").
Remediation: run `npm run index:prewarm`, then re-verify dense_state via index_stats.
```

## Guardrails

- Stay read-only. Do not create caches, rebuild indices, or edit files.
- Treat the current BM25+dense hybrid as the baseline. Do not propose advanced
  RAG changes (cross-encoder re-ranking, semantic chunking, query expansion)
  through this skill — route those to the Layer 7+ plan at
  `plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md`.
- Do not assume the advanced RAG system is implemented. The current system is
  BM25 + all-MiniLM-L6-v2 dense embeddings fused via Reciprocal Rank Fusion
  (RRF, k=60), backed by Turso native vectors (`F8_BLOB` quantization with a
  DiskANN ANN index).
- When `dense_state` is `cold` or `model-only`, the correct action is
  `npm run index:prewarm`. If prewarm fails, classify as a cache issue and
  prepare handoff — do not attempt manual model downloads or Turso database
  surgery.
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
