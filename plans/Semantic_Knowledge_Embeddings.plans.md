# Semantic Knowledge Embeddings (Repo Cortex — Layer 5)

**Status:** [PLANNED]

> Final advanced layer of the Repo Cortex system. Adds ONNX-backed local dense embeddings
> and hybrid BM25 + dense ranking so context retrieval benefits from semantic similarity
> rather than keyword overlap alone.
> This is deliberately the last Repo Cortex layer; Layers 1–3 must be [DONE] before this begins.

## Purpose

The BM25 full-text index in Layer 1 excels at exact keyword and phrase matching, but fails on
paraphrased queries where the user's vocabulary does not overlap with the corpus vocabulary.
Dense embeddings produced by a locally cached ONNX sentence-transformer model close this gap.

**Key design choices:**

- **Local-only**: no external embedding APIs, no network calls at query time. The ONNX model
  is cached locally in `scripts/semantic-index/models/`.
- **Opt-in by default**: BM25 search remains the default. Dense re-ranking is an explicit opt-in
  flag (`--dense`, `use_dense: true` in MCP tool calls) until evaluation proves consistent
  improvement across the query eval set.
- **Hybrid ranking**: when enabled, BM25 and dense scores are linearly combined:
  `score = alpha * bm25_norm + (1-alpha) * cosine_sim` where `alpha` is configurable (default 0.5).
- **Vector storage**: embeddings are stored in a separate `data/embeddings.sqlite` (or as an
  additional table in the main DB), using the `sqlite-vec` extension or a manual blob column.

## Non-goals

- External embedding API calls (OpenAI, Cohere, etc.).
- Training or fine-tuning the embedding model.
- Real-time browser-side embedding (the browser snapshot remains pre-indexed BM25 only;
  dense search is server/Node-side only).
- Changes to `src/` library code.
- NeatChat-internal memory (separate plan: `NeatChat_Local_Retrieval_Memory.plans.md`).

## Dependencies

- [Semantic_Knowledge_Foundation.plans.md](Semantic_Knowledge_Foundation.plans.md) [PLANNED] —
  SQLite corpus index must be [DONE].
- [Semantic_Knowledge_MCP_Tools.plans.md](Semantic_Knowledge_MCP_Tools.plans.md) [PLANNED] —
  MCP tools must be [DONE] so dense re-ranking can be toggled via existing `search_corpus` tool.
- ONNX Runtime for Node (`onnxruntime-node`) — confirm availability or add to devDependencies.
- A sentence-transformer ONNX model (e.g., `all-MiniLM-L6-v2`) downloaded to the local model cache.
- `sqlite-vec` or manual blob embedding storage in SQLite.

## Scope

### ONNX model cache

| Path | Notes |
|---|---|
| `scripts/semantic-index/models/` | Local model cache directory (gitignored) |
| `scripts/semantic-index/download-model.mjs` | Downloads and validates the ONNX model on first use |
| `.gitignore` addition | `scripts/semantic-index/models/` |

### Embedding index

| Artifact | Path | Notes |
|---|---|---|
| Embedding builder | `scripts/semantic-index/embed-index.mjs` | Reads chunks → runs ONNX → stores vectors |
| Embedding storage | `data/embeddings.sqlite` (or main DB `embeddings` table) | Gitignored |
| Hybrid ranker | `scripts/semantic-index/hybrid-rank.mjs` | BM25 + cosine score linear combination |
| Dense query CLI | `scripts/semantic-index/query-dense.mjs` | CLI: `--query`, `--dense`, `--alpha`, `--json` |
| Embedding validation | `scripts/semantic-index/validate-embeddings.mjs` | Assert vector count matches chunk count |
| Eval query set | `scripts/semantic-index/eval-queries.json` | 20 canonical queries + expected top results |
| Eval runner | `scripts/semantic-index/eval-embeddings.mjs` | Runs eval queries; reports MRR@5 for BM25 vs hybrid |

### `eval-queries.json` format

```jsonc
[
  {
    "query": "how does NEAT crossover work",
    "expected_doc_families": ["readme", "plan"],
    "expected_heading_contains": "crossover"
  },
  {
    "query": "worker pool evaluation ordered results",
    "expected_doc_families": ["readme", "plan"],
    "expected_heading_contains": "worker"
  }
]
```

### Hybrid rank formula

$$
\text{score}(d, q) = \alpha \cdot \text{BM25\_norm}(d, q) + (1 - \alpha) \cdot \text{cosine}(\mathbf{e}_d, \mathbf{e}_q)
$$

where $\text{BM25\_norm}$ is min-max normalized over the candidate set and $\mathbf{e}_d$,
$\mathbf{e}_q$ are L2-normalized dense embedding vectors.

### MCP tool extension

The existing `search_corpus` tool in `neataptic-cortex-mcp` gains an optional `use_dense` parameter:

```json
{ "tool": "search_corpus", "args": { "query": "...", "use_dense": true, "alpha": 0.5 } }
```

When `use_dense: false` (default), behavior is identical to the BM25-only path.

### `package.json` scripts

| Script | Command |
|---|---|
| `index:embed` | `node scripts/semantic-index/embed-index.mjs` |
| `index:validate-embeddings` | `node scripts/semantic-index/validate-embeddings.mjs --json` |
| `index:eval` | `node scripts/semantic-index/eval-embeddings.mjs --json` |
| `index:download-model` | `node scripts/semantic-index/download-model.mjs` |

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
source_of_truth: "plans/Semantic_Knowledge_Embeddings.plans.md"
skills: "tracker-handoff, plan-sync-validation, onnx-work"
gate: "semantic-mcp-tools-done"
gate_check: "node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json"
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Semantic_Knowledge_Embeddings.plans.md
```

**Step objective:** Confirm Layers 1–3 are [DONE]. Confirm `onnxruntime-node` availability
in `package.json`. Evaluate `sqlite-vec` extension compatibility with the existing
`better-sqlite3` setup. Select ONNX model (recommend `all-MiniLM-L6-v2` for compact size and
strong sentence similarity). Author remaining step packets.

### Phase 2 — Research [PLANNED]

#### Step 02 — Recon ONNX runtime and SQLite vec storage options (02-researching) [PLANNED]

```yaml
phase: 2
step: 2
agent: "02-researching"
agent_file: ".github/agents/02-researching.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Embeddings.plans.md"
skills: "onnx-work, plan-alignment"
validation:
  - Read src/architecture/network/onnx/ for existing onnxruntime-node usage patterns
  - Confirm sqlite-vec or blob-column approach for vector storage
  - Confirm model download mechanism (e.g., huggingface/onnx-community hub)
```

**Step objective:** Map existing ONNX runtime usage in the repo (e.g., in
`src/architecture/network/onnx/`), confirm the embedding model size and download URL, and
choose the vector storage approach (prefer `sqlite-vec` if compatible with `better-sqlite3`,
else use a `BLOB` column with manual cosine computation).

### Phase 3 — Red tests [PLANNED]

#### Step 03 — Red contracts for embedding builder and hybrid ranker (03-red-testing) [PLANNED]

```yaml
phase: 3
step: 3
agent: "03-red-testing"
agent_file: ".github/agents/03-red-testing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Embeddings.plans.md"
skills: "red-test-contracts, onnx-work"
validation:
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index/embed
```

**Step objective:** Write failing tests for:
- `embed-index.mjs`: given a seeded mini corpus (3 chunks), produces 3 embedding vectors of expected dimension.
- `hybrid-rank.mjs`: given BM25 scores and cosine scores, produces correctly weighted combined ranks.
- `validate-embeddings.mjs`: fails when embedding count does not match chunk count, passes when equal.
Use a tiny mock ONNX model or a pre-computed fixture to avoid downloading a real model in CI.

### Phase 4 — Implementation [PLANNED]

#### Step 04 — Implement embedding builder, hybrid ranker, eval set (04-implementing) [PLANNED]

```yaml
phase: 4
step: 4
agent: "04-implementing"
agent_file: ".github/agents/04-implementing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Embeddings.plans.md"
skills: "onnx-work, agent-script-tooling"
validation:
  - node scripts/semantic-index/download-model.mjs
  - node scripts/semantic-index/embed-index.mjs --dry-run
  - node scripts/semantic-index/embed-index.mjs
  - node scripts/semantic-index/validate-embeddings.mjs --json
  - node scripts/semantic-index/query-dense.mjs --query "NEAT activation" --json
```

**Step objective:** Implement all artifacts:
1. `download-model.mjs` — fetch ONNX model to `scripts/semantic-index/models/`, verify checksum.
2. `embed-index.mjs` — load model via `onnxruntime-node`, embed all chunks, store vectors.
3. `hybrid-rank.mjs` — BM25 + cosine linear combination with configurable `alpha`.
4. `query-dense.mjs` — CLI with `--query`, `--dense`, `--alpha`, `--json`, `--limit`.
5. `validate-embeddings.mjs` — assert vector count matches chunk count.
6. `eval-queries.json` — 20 canonical queries covering key corpus areas.
7. `eval-embeddings.mjs` — MRR@5 report comparing BM25 vs hybrid for each query.
8. Extend `search_corpus` MCP tool with `use_dense` and `alpha` parameters.
9. Add gitignore entries for `scripts/semantic-index/models/` and `data/embeddings.sqlite`.

### Phase 5 — Green validation [PLANNED]

#### Step 05 — Run eval set and confirm hybrid improvement (05-green-testing) [PLANNED]

```yaml
phase: 5
step: 5
agent: "05-green-testing"
agent_file: ".github/agents/05-green-testing.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Embeddings.plans.md"
skills: "green-validation-gates, onnx-work"
validation:
  - node scripts/semantic-index/validate-embeddings.mjs --json
  - node scripts/semantic-index/eval-embeddings.mjs --json
  - npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/semantic-index/embed
  - Confirm hybrid MRR@5 >= BM25-only MRR@5 on eval set
```

**Step objective:** Run the full eval set against the real corpus. Confirm that hybrid
ranking does not regress MRR@5 relative to BM25-only. If hybrid regresses on specific query
classes, document the failure modes and adjust `alpha` or the eval set. Unit tests all green.

### Phase 6 — Docs [PLANNED]

#### Step 06 — Document embedding model, eval methodology, opt-in policy (06-documenting) [PLANNED]

```yaml
phase: 6
step: 6
agent: "06-documenting"
agent_file: ".github/agents/06-documenting.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Embeddings.plans.md"
skills: "educational-docs"
```

**Step objective:** Update `scripts/semantic-index/README.md` to document the embedding
pipeline, the ONNX model choice, the hybrid rank formula, the eval methodology, and the
opt-in policy (`use_dense: false` by default until eval proves consistent improvement).
Include the MRR@5 results from Step 05.

### Phase 7 — Logging [PLANNED]

#### Step 07 — Session log and opt-in policy decision (07-logging) [PLANNED]

```yaml
phase: 7
step: 7
agent: "07-logging"
agent_file: ".github/agents/07-logging.agent.md"
status: "[PLANNED]"
mode: "sequential"
source_of_truth: "plans/Semantic_Knowledge_Embeddings.plans.md"
skills: "tracker-handoff, summarizing-session-log"
```

**Step objective:** Record eval evidence. Decide whether to flip the default from
`use_dense: false` to `use_dense: true` based on MRR@5 results. Record the decision in the
done-state log. Update this tracker to [DONE].

## Acceptance criteria and validation gates

| Gate | Command | Expected |
|---|---|---|
| Model downloads | `node scripts/semantic-index/download-model.mjs` | Exit 0; model present in `scripts/semantic-index/models/` |
| Embeddings build | `node scripts/semantic-index/embed-index.mjs` | Exit 0; vector count = chunk count |
| Embedding validation | `node scripts/semantic-index/validate-embeddings.mjs --json` | `{ pass: true }` |
| Dense query returns results | `node scripts/semantic-index/query-dense.mjs --query "NEAT" --json` | JSON array ≥ 3 results |
| Eval set runs | `node scripts/semantic-index/eval-embeddings.mjs --json` | JSON with MRR@5 BM25 and hybrid |
| Hybrid does not regress | Eval output | hybrid MRR@5 ≥ BM25-only MRR@5 |
| Unit tests green | `npx jest --testPathPattern=scripts/semantic-index/embed` | All pass |
| Gitignore clean | `git status scripts/semantic-index/models/ data/embeddings.sqlite` | Both ignored |

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Active plan: plans/Semantic_Knowledge_Embeddings.plans.md [PLANNED]

Prerequisites (must all be [DONE] before this plan starts):
  - plans/Semantic_Knowledge_Foundation.plans.md
  - plans/Semantic_Knowledge_MCP_Tools.plans.md
  Verify: node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json

Goal: Add ONNX-backed local dense embeddings and hybrid BM25+dense ranking to the Repo Cortex.

Key artifacts:
  scripts/semantic-index/download-model.mjs           (ONNX model download)
  scripts/semantic-index/models/                      (model cache, gitignored)
  scripts/semantic-index/embed-index.mjs              (embedding builder)
  scripts/semantic-index/hybrid-rank.mjs              (BM25 + cosine combiner)
  scripts/semantic-index/query-dense.mjs              (CLI with --dense flag)
  scripts/semantic-index/eval-queries.json            (20 canonical eval queries)
  scripts/semantic-index/eval-embeddings.mjs          (MRR@5 eval runner)
  data/embeddings.sqlite                              (vector store, gitignored)

Hybrid is opt-in (use_dense: false by default) until eval proves MRR@5 improvement.

Start with Step 01 (01-planning): confirm onnxruntime-node availability and choose
the vector storage approach before authoring remaining step packets.

Plan sync check:
  node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Semantic_Knowledge_Embeddings.plans.md
```
