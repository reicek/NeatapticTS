# Cortex RAG Evaluation Suite Architecture

> Extracted from `plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md` (Step 11) for permanent reference.

Complete design for comprehensive RAG evaluation suite with MRR, nDCG, recall@k, context relevance, faithfulness metrics, baseline measurement, and CI regression gate.

---

#### Step 11 — Design RAG evaluation suite architecture [DONE]

```yaml
phase: 1
step: 11
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[DONE]'
source_of_truth: 'plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
copy_paste: true
next_step: 'Step 12 — Design ANN index architecture'
skills:
  - 'plan-alignment'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md'
  - 'node .github/hooks/workflow-update-sync.mjs --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md --json'
```

**Step objective:** Design comprehensive RAG evaluation:

- Metrics: MRR@k, nDCG@k, recall@k, context relevance, faithfulness, latency
- Query taxonomy: 6 classes × 10+ queries per class = 60+ curated eval queries
- Baseline measurement: current system performance on expanded eval set
- Regression detection: CI gate comparing current vs baseline metrics
- A/B comparison: framework for comparing retrieval strategies side-by-side

---

##### RAG Evaluation Suite Architecture — Complete Design

###### A. Problem Statement

The current evaluation infrastructure has critical gaps that prevent reliable measurement of advanced RAG capabilities:

1. **Only 20 queries, only MRR@5**: The existing `eval-queries.json` contains 20 queries that measure only MRR@5 (Mean Reciprocal Rank at k=5). There is no nDCG@k for graded relevance, no Recall@k for coverage, no latency measurement, and no context relevance or faithfulness assessment. 8/20 hybrid queries score 0 MRR@5 (complete misses), indicating the query set does not adequately cover hard retrieval scenarios.

2. **No query taxonomy**: All 20 queries are treated equally regardless of difficulty. A simple lookup like "Network activate" and a cross-boundary query like "how does Network activation use slab" receive the same weight in the aggregate metric. This makes it impossible to identify whether improvements help hard queries or just easy ones.

3. **No graded relevance**: The current hit criteria are binary — a result either matches (`expected_doc_families` + `expected_heading_contains` OR `expected_symbol_contains`) or doesn't. There is no concept of partially relevant results (same topic but different heading) versus highly relevant results (exact target chunk). nDCG@k requires graded relevance judgments.

4. **No regression detection**: There is no CI gate that compares current metrics against a stored baseline. A retrieval improvement in one area can silently regress another, and no automated check catches it.

5. **No A/B framework**: When comparing BM25-only vs. hybrid vs. hybrid+rerank, there is no standardized way to run both conditions, compute paired metrics, and report statistical significance of differences.

6. **No context-level evaluation**: The Step 05 context assembly and Step 10 `search_context` tool produce assembled context windows, but the current eval only measures ranking quality of individual chunks. There is no evaluation of whether assembled context is relevant, complete, or faithful.

7. **No latency measurement**: The Step 04 cross-encoder and Step 10 `search_advanced` pipeline add latency, but the current eval does not measure end-to-end query latency or track it against budgets.

**Goal**: Design an evaluation suite that addresses all seven gaps, provides a reproducible baseline, enables CI regression detection, and supports A/B comparison of retrieval strategies — without requiring human annotation for the automated metrics.

###### B. Query Taxonomy and Expanded Eval Set

**B.1 Taxonomy classes.**

The 6-class taxonomy from the Step 01 audit is adopted with refined definitions:

| Class            | Definition                                                          | Expected families               | Difficulty  | Alpha default     |
| ---------------- | ------------------------------------------------------------------- | ------------------------------- | ----------- | ----------------- |
| `simple_lookup`  | Direct term or symbol match; single relevant family sufficient      | 1 primary                       | Easy        | 0.7 (BM25-heavy)  |
| `cross_boundary` | Requires context from 2+ families; no single-family answer          | 2+ families                     | Medium      | 0.4 (dense-heavy) |
| `multi_hop`      | Requires chaining across 3+ semantic hops or entity graph traversal | 3+ families or 3+ hops          | Hard        | 0.3 (dense-heavy) |
| `exploratory`    | Broad discovery; many partially relevant results acceptable         | 2+ families, 5+ relevant chunks | Medium      | 0.5 (balanced)    |
| `code_specific`  | Symbol, API, or function lookup in ts-source                        | ts-source primary               | Easy–Medium | 0.6 (BM25-heavy)  |
| `plan_specific`  | Roadmap, architecture, or design document retrieval                 | plan, completed-plan primary    | Medium      | 0.6 (BM25-heavy)  |

**B.2 Query distribution target.**

| Class            | Minimum queries | Target queries | Rationale                                    |
| ---------------- | --------------- | -------------- | -------------------------------------------- |
| `simple_lookup`  | 10              | 12             | Adequate coverage of single-family lookups   |
| `cross_boundary` | 10              | 12             | Adequate coverage of multi-family joins      |
| `multi_hop`      | 8               | 10             | Fewer multi-hop queries exist naturally      |
| `exploratory`    | 8               | 10             | Broad queries need diverse topic coverage    |
| `code_specific`  | 10              | 12             | Large ts-source corpus needs strong coverage |
| `plan_specific`  | 10              | 12             | Plans are critical agent context             |
| **Total**        | **56**          | **68**         | Minimum 56, target 68                        |

The existing 20 queries are preserved and reclassified into the taxonomy. 40–48 new queries are added to reach the target.

**B.3 Query classification method.**

Each query is assigned a taxonomy class and difficulty during curation:

```json
{
  "query_id": "sl-001",
  "query": "how does NEAT crossover work",
  "class": "simple_lookup",
  "difficulty": "easy",
  "expected_doc_families": ["readme", "plan"],
  "expected_heading_contains": "crossover",
  "expected_symbol_contains": null,
  "min_rank_of_hit": 3,
  "relevance_grades": [
    { "heading_path_contains": "crossover", "family": "readme", "grade": 3 },
    { "heading_path_contains": "crossover", "family": "ts-source", "grade": 2 },
    { "heading_path_contains": "NEAT", "family": "readme", "grade": 1 }
  ],
  "expected_chunk_ids": [],
  "notes": ""
}
```

**B.4 Query ID scheme.**

Query IDs use a two-letter class prefix followed by a three-digit number:

| Class            | Prefix | Examples            |
| ---------------- | ------ | ------------------- |
| `simple_lookup`  | `sl`   | sl-001, sl-002, ... |
| `cross_boundary` | `cb`   | cb-001, cb-002, ... |
| `multi_hop`      | `mh`   | mh-001, mh-002, ... |
| `exploratory`    | `ex`   | ex-001, ex-002, ... |
| `code_specific`  | `cs`   | cs-001, cs-002, ... |
| `plan_specific`  | `ps`   | ps-001, ps-002, ... |

**B.5 Graded relevance annotations.**

Each query can include `relevance_grades`: an array of relevance grade rules. A grade of 3 = highly relevant (exact target), 2 = relevant (same topic, different aspect), 1 = marginally relevant (tangentially related), 0 = not relevant. When `relevance_grades` is provided, nDCG@k uses these grades. When absent, binary relevance is used (1 for hit, 0 for miss).

The `expected_chunk_ids` field allows specifying exact chunk IDs that should appear in results, enabling Recall@k computation. When `expected_chunk_ids` is empty, recall is computed from `expected_doc_families` + `expected_heading_contains` matches.

**B.6 Curation protocol.**

1. Start with the existing 20 queries, reclassify, and assign query IDs.
2. For each class, curate queries to fill gaps in topic coverage, family diversity, and difficulty spread.
3. Ensure at least 5 queries per class have `relevance_grades` annotations for nDCG@k computation.
4. Ensure at least 3 queries per class have `expected_chunk_ids` for Recall@k computation.
5. Validate that every query returns at least 1 result in BM25-only mode (no zero-result queries in the eval set).
6. Store the expanded set as `scripts/semantic-index/eval-queries-v2.json`.

###### C. Metrics Definitions

**C.1 MRR@k (Mean Reciprocal Rank at k).**

**Formula**: `MRR@k = (1/|Q|) × Σ_{q∈Q} 1/rank_q` where `rank_q` is the rank of the first relevant result for query `q` within the top-k results, or 0 if no relevant result appears in the top-k.

**Properties**:

- Focuses on the first relevant result (order-sensitive).
- Values range from 0 to 1.
- Used as the primary metric for retrieval quality.
- k = 5 is the standard (matching the existing eval).

**Implementation**: Existing `eval-embeddings.mjs` computes MRR@5. The new eval runner computes MRR@k for k ∈ {1, 3, 5, 10}.

**C.2 nDCG@k (Normalized Discounted Cumulative Gain at k).**

**Formula**:

```
DCG@k = Σ_{i=1}^{k} (2^{rel_i} - 1) / log_2(i + 1)
nDCG@k = DCG@k / IDCG@k
```

Where `rel_i` is the relevance grade of the result at position i, and `IDCG@k` is the ideal DCG@k (results sorted by descending relevance grade).

**Properties**:

- Accounts for graded relevance (not just binary hit/miss).
- Penalizes placing marginally relevant results above highly relevant ones.
- Values range from 0 to 1.
- Requires `relevance_grades` annotations; falls back to binary relevance when grades are absent.

**Implementation**: New function in `eval-runner.mjs`. For queries without `relevance_grades`, uses binary grades: grade 1 for family+heading match, grade 0 otherwise.

**C.3 Recall@k.**

**Formula**: `Recall@k = |relevant_results_in_top_k| / |all_relevant_results|`

Where `all_relevant_results` is determined by `expected_chunk_ids` when available, or by counting all chunks matching `expected_doc_families` + `expected_heading_contains` when `expected_chunk_ids` is empty.

**Properties**:

- Measures coverage: how many of all relevant chunks appear in the top-k.
- Values range from 0 to 1.
- Complements MRR (which measures precision of the first hit) with breadth of coverage.

**Implementation**: When `expected_chunk_ids` is provided, count matches in top-k results. When absent, use `expected_doc_families` + `expected_heading_contains` matching (existing hit criteria) to count relevant results.

**C.4 Context relevance.**

**Definition**: For `search_context` tool evaluation, measures what fraction of the assembled context is relevant to the query.

**Formula**: `context_relevance = |relevant_chunks_in_context| / |total_chunks_in_context|`

A chunk is relevant if it matches `expected_doc_families` + (`expected_heading_contains` OR `expected_symbol_contains`).

**Properties**:

- Measures signal-to-noise ratio in the assembled context.
- Complements ranking metrics by evaluating the assembled output agents receive.
- Only computed for `search_context` conditions, not for raw `search_corpus` conditions.

**C.5 Faithfulness (deferred).**

**Definition**: Whether the assembled context contains information that faithfully supports answering the query, without hallucinated or contradictory content.

**Status**: Faithfulness evaluation requires a language model or human annotator to judge whether the retrieved context actually supports answering the query. This is deferred to Phase 3 (Validation and integration) as a human evaluation on a 20-query subset. The eval infrastructure provides the scaffolding for collecting assembled contexts for later annotation, but automated faithfulness scoring is out of scope for Phase 1.

**C.6 Latency.**

**Definition**: End-to-end wall-clock time from query submission to result return, measured in milliseconds.

**Properties**:

- Reported as P50, P95, and max across all queries.
- Measured per condition (BM25-only, hybrid, hybrid+rerank, search_advanced, search_context).
- Latency budgets from Step 10 Section K.4 serve as regression thresholds.

**C.7 Metric summary.**

| Metric            | Computation                            | Requires                                     | Scope                 | k values      |
| ----------------- | -------------------------------------- | -------------------------------------------- | --------------------- | ------------- |
| MRR@k             | Reciprocal rank of first hit           | `expected_doc_families` + heading/symbol     | All conditions        | 1, 3, 5, 10   |
| nDCG@k            | Normalized DCG with graded relevance   | `relevance_grades` (falls back to binary)    | All conditions        | 5, 10         |
| Recall@k          | Fraction of relevant results in top-k  | `expected_chunk_ids` or family+heading match | All conditions        | 5, 10, 20     |
| Context relevance | Relevant fraction of assembled context | `expected_doc_families` + heading/symbol     | `search_context` only | N/A           |
| Faithfulness      | Human evaluation of answer support     | Assembled context + query                    | 20-query subset       | N/A (Phase 3) |
| Latency           | P50, P95, max wall-clock time          | Timer                                        | All conditions        | N/A           |

###### D. Baseline Measurement Protocol

**D.1 Baseline conditions.**

The baseline is measured under four conditions:

| Condition ID       | Name                    | Retrieval                   | Re-ranking               | Context assembly        | Query expansion        | Metadata filter |
| ------------------ | ----------------------- | --------------------------- | ------------------------ | ----------------------- | ---------------------- | --------------- |
| `bm25_only`        | BM25-only               | BM25 FTS5                   | None                     | None                    | None                   | None            |
| `hybrid`           | Hybrid (BM25 + dense)   | BM25 + dense (α=0.5)        | None                     | None                    | None                   | None            |
| `hybrid_rerank`    | Hybrid + cross-encoder  | BM25 + dense (α=0.5)        | Cross-encoder re-ranking | None                    | None                   | None            |
| `advanced_default` | Full pipeline (default) | Classification-aware hybrid | Cross-encoder when warm  | Budget-managed assembly | Auto by classification | None            |

**D.2 Baseline measurement procedure.**

1. **Rebuild the index**: `node scripts/semantic-index/build-index.mjs --force`
2. **Warm the dense embeddings**: `node scripts/semantic-index/prewarm-dense.mjs`
3. **Warm the reranker** (for `hybrid_rerank` and `advanced_default`): `node scripts/semantic-index/index:prewarm:reranker`
4. **Run the eval**: `node scripts/semantic-index/eval-runner.mjs --query-file scripts/semantic-index/eval-queries-v2.json --condition all --json`
5. **Store baseline**: The eval runner writes results to `data/eval-baselines/baseline-<timestamp>.json`

**D.3 Baseline storage format.**

```json
{
  "baseline_id": "baseline-2026-06-08T18:00:00Z",
  "timestamp": "2026-06-08T18:00:00Z",
  "corpus_stats": {
    "total_documents": 1408,
    "total_chunks": 31648,
    "total_families": 10,
    "index_build_timestamp": "2026-06-08T17:00:00Z"
  },
  "conditions": {
    "bm25_only": {
      "mrr_at_1": 0.0,
      "mrr_at_3": 0.0,
      "mrr_at_5": 0.225,
      "mrr_at_10": 0.0,
      "ndcg_at_5": 0.0,
      "ndcg_at_10": 0.0,
      "recall_at_5": 0.0,
      "recall_at_10": 0.0,
      "recall_at_20": 0.0,
      "latency_ms": { "p50": 0, "p95": 0, "max": 0 },
      "per_query": [
        {
          "query_id": "sl-001",
          "mrr_at_5": 0.0,
          "ndcg_at_5": 0.0,
          "recall_at_5": 0.0,
          "latency_ms": 0
        }
      ]
    },
    "hybrid": { "...": "..." },
    "hybrid_rerank": { "...": "..." },
    "advanced_default": { "...": "..." }
  },
  "per_class": {
    "simple_lookup": { "mrr_at_5": 0.0, "ndcg_at_5": 0.0, "recall_at_5": 0.0 },
    "cross_boundary": { "mrr_at_5": 0.0, "ndcg_at_5": 0.0, "recall_at_5": 0.0 },
    "multi_hop": { "mrr_at_5": 0.0, "ndcg_at_5": 0.0, "recall_at_5": 0.0 },
    "exploratory": { "mrr_at_5": 0.0, "ndcg_at_5": 0.0, "recall_at_5": 0.0 },
    "code_specific": { "mrr_at_5": 0.0, "ndcg_at_5": 0.0, "recall_at_5": 0.0 },
    "plan_specific": { "mrr_at_5": 0.0, "ndcg_at_5": 0.0, "recall_at_5": 0.0 }
  }
}
```

**D.4 Initial baseline values (from existing 20-query eval).**

These values are from the Step 01 audit and serve as reference points only. The actual baseline will be computed on the expanded 56+ query set:

| Metric      | BM25-only (20 queries) | Hybrid (20 queries) |
| ----------- | ---------------------- | ------------------- |
| MRR@5       | 0.225                  | 0.308               |
| nDCG@5      | Not measured           | Not measured        |
| Recall@5    | Not measured           | Not measured        |
| Latency P50 | Not measured           | ~50–80 ms           |

**D.5 Per-class baseline.**

Per-class baselines are computed by grouping queries by their `class` field and computing metrics within each group. This identifies which query classes are underserved and whether improvements help hard classes or only easy ones.

###### E. Regression Detection

**E.1 CI gate design.**

The eval regression gate runs during CI after index build and dense warmup:

```bash
node scripts/semantic-index/eval-runner.mjs \
  --query-file scripts/semantic-index/eval-queries-v2.json \
  --condition all \
  --baseline data/eval-baselines/baseline-latest.json \
  --regression-threshold 0.01 \
  --json
```

**E.2 Regression thresholds.**

| Metric               | Absolute threshold | Regression delta                | Action    |
| -------------------- | ------------------ | ------------------------------- | --------- |
| MRR@5 (hybrid)       | ≥ 0.25             | < baseline − 0.01               | FAIL gate |
| nDCG@5 (hybrid)      | ≥ 0.40             | < baseline − 0.01               | WARN gate |
| Recall@5 (hybrid)    | ≥ 0.45             | < baseline − 0.01               | WARN gate |
| MRR@5 (bm25_only)    | ≥ 0.20             | < baseline − 0.01               | WARN gate |
| Latency P95 (hybrid) | ≤ 500 ms           | > baseline + 100 ms             | WARN gate |
| MRR@5 per class      | N/A                | < baseline − 0.03 for any class | WARN gate |

**FAIL** means the gate exits non-zero and blocks merging. **WARN** means the gate reports the regression but does not block merging.

**E.3 Baseline update protocol.**

1. When a deliberate improvement is merged that changes baseline metrics, run the full eval suite on the updated codebase.
2. If the new metrics meet or exceed the previous baseline for all FAIL thresholds, copy the new baseline to `data/eval-baselines/baseline-latest.json`.
3. Commit the updated baseline alongside the code change that caused the improvement.
4. Never update the baseline without a corresponding code or query-set change.

**E.4 Gate output format.**

```json
{
  "gate": "eval-regression",
  "pass": true,
  "evidence": {
    "condition": "hybrid",
    "metrics": {
      "mrr_at_5": {
        "current": 0.33,
        "baseline": 0.32,
        "delta": "+0.010",
        "status": "PASS"
      },
      "ndcg_at_5": {
        "current": 0.45,
        "baseline": 0.44,
        "delta": "+0.010",
        "status": "PASS"
      },
      "recall_at_5": {
        "current": 0.52,
        "baseline": 0.51,
        "delta": "+0.010",
        "status": "PASS"
      }
    },
    "per_class": {
      "simple_lookup": {
        "mrr_at_5": {
          "current": 0.55,
          "baseline": 0.52,
          "delta": "+0.03",
          "status": "PASS"
        }
      },
      "cross_boundary": {
        "mrr_at_5": {
          "current": 0.28,
          "baseline": 0.27,
          "delta": "+0.01",
          "status": "PASS"
        }
      }
    },
    "regressions": [],
    "warnings": []
  },
  "fix_hint": null
}
```

###### F. A/B Comparison Framework

**F.1 Side-by-side comparison.**

The eval runner supports comparing two retrieval configurations against the same query set:

```bash
node scripts/semantic-index/eval-runner.mjs \
  --query-file scripts/semantic-index/eval-queries-v2.json \
  --condition hybrid \
  --condition hybrid_rerank \
  --compare \
  --json
```

**F.2 Comparison output format.**

```json
{
  "comparison_id": "comparison-2026-06-08T18:00:00Z",
  "condition_a": "hybrid",
  "condition_b": "hybrid_rerank",
  "query_count": 68,
  "metrics": {
    "mrr_at_5": {
      "a": 0.32,
      "b": 0.38,
      "delta": "+0.060",
      "significant": true,
      "p_value": 0.012
    },
    "ndcg_at_5": {
      "a": 0.44,
      "b": 0.49,
      "delta": "+0.050",
      "significant": true,
      "p_value": 0.023
    }
  },
  "per_query_wins": {
    "a_wins": 12,
    "b_wins": 38,
    "ties": 18
  },
  "per_class_delta": {
    "simple_lookup": { "mrr_at_5_delta": "+0.02" },
    "cross_boundary": { "mrr_at_5_delta": "+0.08" }
  }
}
```

**F.3 Statistical significance.**

Per-query paired comparison uses the Wilcoxon signed-rank test (non-parametric, appropriate for MRR/Recall which are not normally distributed). A difference is reported as `significant: true` when p < 0.05.

**F.4 Alpha sweep.**

The eval runner supports alpha sweep mode to find the optimal blend weight:

```bash
node scripts/semantic-index/eval-runner.mjs \
  --query-file scripts/semantic-index/eval-queries-v2.json \
  --condition hybrid \
  --alpha-sweep 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --json
```

Output includes MRR@5, nDCG@5, and Recall@5 for each alpha value, plus the per-class breakdown. The sweep identifies the optimal alpha for each query class.

**F.5 Classification-aware comparison.**

When the `search_advanced` pipeline is active, the A/B framework can compare classification-aware retrieval against fixed-alpha hybrid:

```bash
node scripts/semantic-index/eval-runner.mjs \
  --query-file scripts/semantic-index/eval-queries-v2.json \
  --condition hybrid \
  --condition advanced_default \
  --compare \
  --json
```

This produces a comparison where condition A uses fixed α=0.5 and condition B uses classification-derived alpha and expansion per the Step 03 routing table.

###### G. Eval Runner Architecture

**G.1 Module structure.**

```
scripts/semantic-index/
  eval-queries-v2.json        ← Expanded query set (56–68 queries with taxonomy)
  eval-runner.mjs              ← Unified eval runner (replaces eval-embeddings.mjs)
  eval-metrics.mjs             ← Metric computation functions (MRR, nDCG, Recall)
  eval-compare.mjs             ← A/B comparison and statistical tests
  eval-baseline.mjs            ← Baseline storage and regression detection
```

The existing `eval-embeddings.mjs` is preserved for backward compatibility. The new `eval-runner.mjs` subsumes its functionality and extends it with the full metric suite.

**G.2 `eval-runner.mjs` CLI interface.**

```
node scripts/semantic-index/eval-runner.mjs [options]

Options:
  --query-file <path>           Path to eval queries JSON file
                                 Default: scripts/semantic-index/eval-queries-v2.json
  --condition <name>            Eval condition(s): bm25_only, hybrid, hybrid_rerank,
                                 advanced_default, all
                                 Default: all
  --alpha <n>                   Override hybrid alpha (default: 0.5)
  --limit <n>                   Results per query (default: 10)
  --baseline <path>             Path to baseline JSON for regression detection
  --regression-threshold <n>    MRR@5 regression threshold (default: 0.01)
  --compare                     Run A/B comparison between two conditions
  --alpha-sweep <values>        Comma-separated alpha values to sweep
  --database <path>             Override corpus database path
  --embeddings-database <path>  Override embeddings database path
  --model-directory <path>      Override model cache directory
  --model-id <id>               Override model identifier
  --reranker-model <id>         Override reranker model identifier
  --output <path>               Write full results to file (JSON)
  --json                        Emit JSON summary to stdout
  --help                        Show help and exit
```

**G.3 `eval-metrics.mjs` exported functions.**

```typescript
// MRR@k: Mean Reciprocal Rank at k
export function computeMRR(results, querySpec, k = 5): number;

// nDCG@k: Normalized Discounted Cumulative Gain at k
export function computeNDCG(results, querySpec, k = 5): number;

// Recall@k: Fraction of relevant results in top-k
export function computeRecall(
  results,
  querySpec,
  allRelevantCount,
  k = 5,
): number;

// Context relevance: Fraction of relevant chunks in assembled context
export function computeContextRelevance(assembledChunks, querySpec): number;

// Aggregate metrics across all queries
export function aggregateMetrics(
  perQueryResults,
  kValues = [1, 3, 5, 10],
): AggregateMetrics;

// Per-class aggregation
export function aggregateByClass(
  perQueryResults,
  kValues = [1, 3, 5, 10],
): Record<string, AggregateMetrics>;
```

**G.4 `eval-compare.mjs` exported functions.**

```typescript
// Wilcoxon signed-rank test for paired per-query differences
export function wilcoxonSignedRank(
  valuesA,
  valuesB,
): { statistic: number; pValue: number; significant: boolean };

// Compare two conditions
export function compareConditions(
  conditionA,
  conditionB,
  perQueryResultsA,
  perQueryResultsB,
): ComparisonResult;

// Alpha sweep: run hybrid with multiple alpha values
export function alphaSweep(
  querySpecs,
  alphaValues,
  options,
): Record<number, AggregateMetrics>;
```

**G.5 `eval-baseline.mjs` exported functions.**

```typescript
// Load baseline from file
export function loadBaseline(baselinePath): BaselineData;

// Save baseline to file
export function saveBaseline(baselinePath, data): void;

// Check regression: compare current metrics against baseline
export function checkRegression(
  current,
  baseline,
  thresholds,
): RegressionCheckResult;
```

**G.6 Condition execution.**

Each condition maps to a retrieval configuration:

| Condition          | `search_corpus` params                                   | Post-retrieval                                                  |
| ------------------ | -------------------------------------------------------- | --------------------------------------------------------------- |
| `bm25_only`        | `{ query, family, limit, alpha: 1.0, use_dense: false }` | None                                                            |
| `hybrid`           | `{ query, family, limit, alpha: 0.5, use_dense: true }`  | None                                                            |
| `hybrid_rerank`    | `{ query, family, limit, alpha: 0.5, use_dense: true }`  | Cross-encoder re-ranking (top `rerank_candidates`)              |
| `advanced_default` | `{ query, family, limit }` via `search_advanced`         | Full pipeline: classify → expand → retrieve → rerank → assemble |

The `advanced_default` condition requires the full pipeline from Step 10 to be available. If the pipeline is not yet implemented, the eval runner skips this condition with a clear warning.

**G.7 Latency measurement.**

Each condition measures wall-clock time from query submission to result return using `performance.now()`. The timer includes:

- Query embedding computation (for dense conditions)
- BM25 search
- Dense candidate retrieval and cosine scoring
- Hybrid ranking
- Cross-encoder re-ranking (for `hybrid_rerank` and `advanced_default`)
- Context assembly (for `advanced_default`)

Latency is reported per-query and aggregated as P50, P95, and max.

**G.8 Query execution order.**

Queries are executed in taxonomy class order (simple_lookup → cross_boundary → multi_hop → exploratory → code_specific → plan_specific) to ensure deterministic per-class aggregation. Within each class, queries are executed in query_id order.

The dense embedder is initialized once before all queries and released after all queries complete (matching the existing `eval-embeddings.mjs` pattern).

###### H. Query Schema v2

**H.1 Schema definition.**

The expanded query set uses a v2 schema that extends the existing v1 schema with taxonomy, graded relevance, and chunk-level annotations:

```typescript
interface EvalQueryV2 {
  query_id: string; // e.g., "sl-001", "cb-005", "mh-002"
  query: string; // Natural language query text
  class: EvalQueryClass; // Taxonomy class
  difficulty: 'easy' | 'medium' | 'hard';
  expected_doc_families: string[]; // Families that contain relevant results
  expected_heading_contains: string | null; // Heading path substring
  expected_symbol_contains: string | null; // Symbol name substring
  min_rank_of_hit: number; // Maximum acceptable rank for first hit
  relevance_grades?: Array<{
    // Optional graded relevance annotations
    heading_path_contains: string;
    family: string;
    grade: 0 | 1 | 2 | 3; // 0=not relevant, 1=marginal, 2=relevant, 3=highly relevant
  }>;
  expected_chunk_ids?: number[]; // Optional exact chunk IDs for recall computation
  metadata_filter?: object; // Optional metadata filter for code_specific queries
  notes?: string; // Curation notes
}

type EvalQueryClass =
  | 'simple_lookup'
  | 'cross_boundary'
  | 'multi_hop'
  | 'exploratory'
  | 'code_specific'
  | 'plan_specific';
```

**H.2 Backward compatibility.**

The v2 schema is a superset of v1. The eval runner accepts both v1 and v2 query files:

- v1 queries (without `query_id`, `class`, `difficulty`) are assigned `class: 'simple_lookup'`, `difficulty: 'medium'`, and `query_id: 'v1-NNN'`.
- v1 queries without `relevance_grades` or `expected_chunk_ids` use binary relevance and family+heading matching for nDCG and Recall computation.

**H.3 v1-to-v2 migration.**

A migration script (`scripts/semantic-index/migrate-eval-queries.mjs`) converts the existing v1 file to v2 format:

1. Assigns query IDs based on best-fit taxonomy class.
2. Sets `difficulty` based on `min_rank_of_hit` (≤3 → easy, ≤5 → medium, >5 → hard).
3. Preserves all existing fields unchanged.
4. Leaves `relevance_grades` and `expected_chunk_ids` empty for manual curation.

###### I. Eval Output Format

**I.1 Standard output format.**

When `--json` is passed, the eval runner emits a JSON summary:

```json
{
  "eval_id": "eval-2026-06-08T18:00:00Z",
  "query_file": "scripts/semantic-index/eval-queries-v2.json",
  "query_count": 68,
  "conditions": ["bm25_only", "hybrid", "hybrid_rerank", "advanced_default"],
  "results": {
    "bm25_only": {
      "mrr_at_1": 0.0,
      "mrr_at_3": 0.0,
      "mrr_at_5": 0.225,
      "mrr_at_10": 0.0,
      "ndcg_at_5": 0.0,
      "ndcg_at_10": 0.0,
      "recall_at_5": 0.0,
      "recall_at_10": 0.0,
      "recall_at_20": 0.0,
      "latency_ms": { "p50": 0, "p95": 0, "max": 0 },
      "zero_hit_queries": 0,
      "per_class": {
        "simple_lookup": {
          "mrr_at_5": 0.0,
          "ndcg_at_5": 0.0,
          "recall_at_5": 0.0
        },
        "cross_boundary": {
          "mrr_at_5": 0.0,
          "ndcg_at_5": 0.0,
          "recall_at_5": 0.0
        }
      }
    },
    "hybrid": { "...": "..." },
    "hybrid_rerank": { "...": "..." },
    "advanced_default": { "...": "..." }
  },
  "regression": {
    "checked": true,
    "baseline": "data/eval-baselines/baseline-latest.json",
    "threshold": 0.01,
    "failures": [],
    "warnings": []
  }
}
```

**I.2 Human-readable output.**

When `--json` is not passed, the eval runner emits a formatted table:

```
Eval Results (68 queries)

Condition         MRR@5  nDCG@5  Recall@5  Latency P50  Zero-hit
─────────────────────────────────────────────────────────────────
bm25_only         0.225  0.380   0.450     25ms         4
hybrid            0.308  0.440   0.520     55ms         2
hybrid_rerank     0.380  0.490   0.560     310ms        1
advanced_default  0.420  0.530   0.600     480ms        0

Per-class MRR@5:
                  sl     cb     mh     ex     cs     ps
bm25_only         0.55   0.15   0.08   0.30   0.28   0.18
hybrid            0.60   0.22   0.12   0.35   0.35   0.25
hybrid_rerank     0.65   0.30   0.18   0.38   0.42   0.32
advanced_default  0.68   0.38   0.25   0.42   0.45   0.38

Regression: PASS (0 failures, 0 warnings)
```

**I.3 Full output file.**

When `--output <path>` is passed, the eval runner writes the complete per-query detail to a JSON file, including latency per query, ranked results, and hit positions. This file is used for debugging, A/B comparison input, and historical tracking.

###### J. Implementation File Map

**J.1 New files.**

| File                                              | Purpose                                                                          | Approximate size |
| ------------------------------------------------- | -------------------------------------------------------------------------------- | ---------------- |
| `scripts/semantic-index/eval-queries-v2.json`     | Expanded query set (56–68 queries with taxonomy and graded relevance)            | ~300 lines       |
| `scripts/semantic-index/eval-runner.mjs`          | Unified eval runner (CLI, condition execution, output formatting)                | ~400 lines       |
| `scripts/semantic-index/eval-metrics.mjs`         | Metric computation functions (MRR, nDCG, Recall, context relevance)              | ~250 lines       |
| `scripts/semantic-index/eval-compare.mjs`         | A/B comparison, Wilcoxon test, alpha sweep                                       | ~200 lines       |
| `scripts/semantic-index/eval-baseline.mjs`        | Baseline storage, regression detection, gate logic                               | ~150 lines       |
| `scripts/semantic-index/migrate-eval-queries.mjs` | v1 → v2 query schema migration script                                            | ~80 lines        |
| `data/eval-baselines/.gitkeep`                    | Directory for baseline JSON files (baselines are not committed; generated by CI) | 0 lines          |

**J.2 Modified files.**

| File                                         | Change                                                                                                           | Scope          |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------- |
| `scripts/semantic-index/eval-embeddings.mjs` | Add deprecation notice in JSDoc and CLI help text pointing to `eval-runner.mjs`; preserve backward compatibility | ~5 lines added |
| `package.json`                               | Add `eval:rag`, `eval:rag:compare`, `eval:rag:regression`, `eval:rag:sweep` scripts                              | ~8 lines added |

**J.3 Package script additions.**

```json
{
  "scripts": {
    "eval:rag": "node scripts/semantic-index/eval-runner.mjs --json",
    "eval:rag:compare": "node scripts/semantic-index/eval-runner.mjs --condition hybrid --condition hybrid_rerank --compare --json",
    "eval:rag:regression": "node scripts/semantic-index/eval-runner.mjs --baseline data/eval-baselines/baseline-latest.json --regression-threshold 0.01 --json",
    "eval:rag:sweep": "node scripts/semantic-index/eval-runner.mjs --condition hybrid --alpha-sweep 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 --json"
  }
}
```

**J.4 Baseline file management.**

Baselines are stored in `data/eval-baselines/` (gitignored). The CI pipeline:

1. Runs `eval:rag:regression` after `index:prewarm` to check against the stored baseline.
2. On merge to main, runs `eval:rag` with `--output` to generate a new baseline.
3. Commits `data/eval-baselines/baseline-latest.json` when the baseline is intentionally updated.

###### K. Eval Design for the Eval Suite Itself

**K.1 Eval-runner self-test queries.**

The eval runner itself needs validation. A small set of 5 self-test queries validates that the metric computation is correct:

| Query ID       | Query                               | Condition | Expected MRR@5 | Expected behavior         |
| -------------- | ----------------------------------- | --------- | -------------- | ------------------------- |
| `selftest-001` | "Network activate" (hits at rank 1) | bm25_only | 1.0            | First result matches      |
| `selftest-002` | "Network activate" (hits at rank 3) | bm25_only | 0.333          | Third result matches      |
| `selftest-003` | "completely nonexistent query xyz"  | bm25_only | 0.0            | No results match          |
| `selftest-004` | "NEAT speciation" (multi-hit)       | hybrid    | ≥ 0.2          | At least one hit in top 5 |
| `selftest-005` | "mutation" (many partial matches)   | hybrid    | ≥ 0.1          | At least one hit          |

These self-test queries are stored in `scripts/semantic-index/eval-queries-selftest.json` and run as a smoke test before the full eval suite.

**K.2 Regression thresholds for the eval runner.**

| Check                     | Threshold                              | Rationale                                            |
| ------------------------- | -------------------------------------- | ---------------------------------------------------- |
| Self-test MRR@5 pass rate | 5/5 (100%)                             | All self-test queries must produce expected results  |
| Baseline file parse       | Must parse without error               | Corrupt baseline blocks regression detection         |
| Condition execution       | All requested conditions must complete | Missing conditions produce incomplete results        |
| Metric range              | 0.0 ≤ MRR/nDCG/Recall ≤ 1.0            | Metrics must be in valid range                       |
| Zero-hit count            | < 30% of queries per condition         | Excessive zero-hit queries indicate broken retrieval |

**K.3 Integration with Phase 3 validation.**

Phase 3 (Validation and integration) will:

1. Run the full eval suite on the completed advanced RAG pipeline.
2. Produce the authoritative baseline `baseline-latest.json`.
3. Execute the regression gate as a CI check.
4. Conduct human evaluation of context relevance and faithfulness on a 20-query subset.
5. Validate that all tool-specific eval queries from Steps 04–10 pass their regression thresholds.

---

_Source: `plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md` lines 7558–8308 (Step 11)_
