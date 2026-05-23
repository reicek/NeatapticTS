# Semantic Knowledge Embeddings Log

**Status:** [DONE]

## Repo Cortex Layer 5 closeout

- [DONE] Delivered local ONNX dense embeddings for Repo Cortex with a BLOB-first `data/embeddings.sqlite` vector store, incremental embedding by `chunk_sha256` + `model_id`, and a gitignored local model cache under `scripts/semantic-index/models/`.
- [DONE] Added the `ts-source` corpus family through `ts-chunker.mjs`, extending the corpus with TypeScript/JSDoc symbol-level chunks while preserving BM25 as the default search path.
- [DONE] Added hybrid BM25 + dense ranking, `query-dense.mjs`, `eval-embeddings.mjs`, `validate-embeddings.mjs`, and `cortex-embeddings.gate.mjs`; MCP `search_corpus` accepts opt-in `use_dense` plus `alpha`, and `scan_code_quality` exposes the code-quality scanner.
- [DONE] Evaluation evidence: `validate-embeddings.mjs --json` passed with 29,315 chunks and 29,315 embeddings; `eval-embeddings.mjs --json` passed with BM25 MRR@5 `0.1875`, hybrid MRR@5 `0.300`, improvement `+0.1125`, required minimum `+0.02`, query count `20`, and `tsSourceQueries: 16`; `cortex-embeddings.gate.mjs --json` passed.
- [DONE] Focused tests stayed green: embed (`4` suites / `6` tests), ts-chunk (`1` suite / `1` test), and code-quality (`1` suite / `1` test`).
- [DONE] Read-only hybrid top-5 family probe found at least one `ts-source` result in 10 of 20 eval queries, exceeding the required 5-query floor.
- [DONE] Documentation closeout: `scripts/semantic-index/README.md` documents ONNX model/assets, `chunk_embeddings` schema, incremental embedding, hybrid rank formula, opt-in policy, MRR@5 results, 10-family corpus table, dense query examples, generated DB expectations, and validation commands; `npm run docs` exited 0 with 176 Mermaid diagrams.
- [DONE] Script docs closeout: all 11 `.mjs` scripts in scope now carry `@description`, `@param`, and `@returns` JSDoc; `--help` output was spot-checked on `build-index.mjs` and `cortex-embeddings.gate.mjs`.
- [DONE] Policy decision: keep `use_dense` default `false`. The +0.1125 MRR@5 improvement proves warm hybrid ranking value, but default-on dense is deferred because the embedding DB and model cache are generated local artifacts and need the Layer 6 prewarm/readiness/degradation contract before default-on behavior is safe.
- [DONE] Post-restart closure evidence: `npm ci` passed after VS restart, so the earlier Windows `better_sqlite3.node` file-lock blocker did not recur and is no longer an open closure caveat.

## Closure validation

- [DONE] `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/completed/Semantic_Knowledge_Embeddings.plans.md` passed with 0 errors and 0 warnings after archive.
- [DONE] `node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json` passed.
- [DONE] Archive paths confirmed under `plans/completed/` for both `Semantic_Knowledge_Embeddings.plans.md` and `Semantic_Knowledge_Embeddings.logs.md`.
- [DONE] Duplicate active root tracker `plans/Semantic_Knowledge_Embeddings.plans.md` was removed after confirming the compressed archive/log pair and existing plan index/roadmap entries already pointed at the completed archive.

## Residual risks

- `data/embeddings.sqlite` and `scripts/semantic-index/models/` remain generated local artifacts; fresh clones are cold until the operator downloads the model and builds embeddings.
- Default-on dense search is intentionally deferred to `plans/Semantic_Knowledge_Dense_Prewarm.plans.md`, which owns readiness states, prewarm bootstrap, and graceful MCP degradation.
