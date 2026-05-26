# Semantic Knowledge Embeddings (Repo Cortex - Layer 5)

**Status:** [DONE]

## Scope

Closed Repo Cortex Layer 5 baseline for local ONNX dense embeddings and hybrid BM25 + dense ranking. This layer adds a TypeScript/JSDoc `ts-source` corpus family, BLOB-first `data/embeddings.sqlite` vector storage, local model download and validation scripts, hybrid ranking/eval/gate scripts, and opt-in MCP dense retrieval.

Out of scope: committing generated model/cache artifacts, browser-side dense search, NeatChat conversational memory, `src/` library behavior changes, and flipping the MCP default to `use_dense: true`.

## Final state

- [DONE] Repo Cortex Layer 1 and Layer 2 prerequisites were confirmed through the archived foundation/tool trackers and the `cortex-mcp-smoke` gate.
- [DONE] Research selected the BLOB-first vector-store path and treated `sqlite-vec` as a conditional future upgrade until Windows/CI extension loading is proven.
- [DONE] Red contracts covered embedding storage, incremental re-embedding, hybrid ranking, embedding validation, TS/JSDoc chunking, code-quality scanning, and the embeddings gate.
- [DONE] Implementation delivered the local ONNX model workflow, `ts-source` corpus extraction, dense embedding builder, hybrid ranker, dense query CLI, eval runner, code-quality scanner, and MCP `search_corpus` / `scan_code_quality` extensions.
- [DONE] Green validation proved the full corpus has matching chunks and embeddings, the hybrid MRR@5 threshold clears the required improvement, and the focused Jest slices remain green.
- [DONE] Documentation recorded the model assets, `chunk_embeddings` schema, incremental embedding rule, hybrid rank formula, opt-in policy, MRR@5 results, corpus family table, and CLI/help contracts.
- [DONE] Logging closed the plan, kept `use_dense` conservative after final post-restart validation, reconciled the duplicate active/completed tracker state, and left the archive/log pair as the sole Layer 5 tracker source of truth.

## Policy decision

`use_dense` remains `false` by default for Layer 5.

The full 20-query eval set cleared the gate by a wide margin: BM25 MRR@5 `0.1875`, hybrid MRR@5 `0.300`, improvement `+0.1125`, with a required minimum improvement of `+0.02`. That evidence proves the hybrid ranker is valuable when embeddings are warm, but it does not make default-on dense safe by itself.

The dense model cache and embedding database are generated local artifacts (`scripts/semantic-index/models/` and `data/embeddings.sqlite`) and are intentionally gitignored. A fresh clone can therefore be cold, and default-on dense needs an explicit prewarm/readiness/degradation contract before callers can treat dense search as the baseline. Post-restart validation did not change that conclusion: `npm ci`, embeddings validation, the eval runner, and the embeddings gate all passed, but none of them remove the cold-clone readiness gap. That operational flip belongs to `plans/Semantic_Knowledge_Dense_Prewarm.plans.md`.

## Audit summary

- `node scripts/semantic-index/validate-embeddings.mjs --json` passed with 29,315 corpus chunks and 29,315 `all-MiniLM-L6-v2` embeddings at dimension 384.
- `node scripts/semantic-index/eval-embeddings.mjs --json` passed with BM25 MRR@5 `0.1875`, hybrid MRR@5 `0.300`, improvement `+0.1125`, query count `20`, and `tsSourceQueries: 16`.
- `node scripts/agent-customization/gates/cortex-embeddings.gate.mjs --json` passed.
- Focused Jest slices passed: embed (`4` suites / `6` tests), ts-chunk (`1` suite / `1` test), and code-quality (`1` suite / `1` test).
- Read-only hybrid top-5 family probe found `ts-source` results in 10 of 20 eval queries, exceeding the required 5-query evidence floor.
- `npm run docs` passed with 176 Mermaid diagrams and regenerated HTML docs.
- Post-restart `npm ci` passed after VS restart, installing 1,152 packages and closing the earlier Windows `better_sqlite3.node` file-lock blocker honestly.
- The duplicate active tracker at `plans/Semantic_Knowledge_Embeddings.plans.md` was retired so `plans/completed/Semantic_Knowledge_Embeddings.plans.md` and its matching log remain the only Layer 5 tracker pair.
- Final archive validation is recorded in the matching audit log.

## Reopen conditions

Reopen this layer only for changes to the embedding schema, model identity or checksum policy, TS/JSDoc chunking contract, hybrid ranking formula, eval threshold, dense MCP parameter contract, or embeddings gate behavior.

Do not reopen this plan just to flip `use_dense` on by default. Default-on dense, prewarm, readiness states, and graceful cold-state degradation belong to `plans/Semantic_Knowledge_Dense_Prewarm.plans.md`.

## Audit log

See `plans/completed/Semantic_Knowledge_Embeddings.logs.md`.
