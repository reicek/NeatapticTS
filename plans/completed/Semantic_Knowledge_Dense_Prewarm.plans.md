# Semantic Knowledge Dense Prewarm (Repo Cortex - Layer 6)

**Status:** [DONE]

## Scope

Closed Repo Cortex Layer 6 baseline for the default-on dense operational contract. This layer adds the idempotent dense prewarm bootstrap, the direct readiness probe, the lightweight readiness gate, the `search_corpus` default flip to `use_dense: true`, and the honest MCP degradation contract for cold and model-only states.

Out of scope: changing Layer 5 embedding quality policy, browser-side dense search, `src/` library behavior, ANN indexing, GPU inference, or NeatChat-local retrieval storage.

## Final state

- [DONE] Phase 1 confirmed the archived Layer 5 prerequisite, verified Layer 5 artifacts and `data/embeddings.sqlite`, and captured that pre-change MCP behavior treated omitted `use_dense` as false via `options.use_dense === true`.
- [DONE] The workflow gap on validation-gate H2 headings was repaired by `00-helping` in `scripts/agent-customization/mcp/mcp-plan-utils.mjs`; the focused MCP regression test passed and the learning event was recorded in `.github/ai-learning/learning-log.jsonl`.
- [DONE] Phase 2 research confirmed `queryDenseIndex` did not own `dense_state`, the MCP schema lacked a `use_dense` default, `scripts/semantic-index/models/` was already gitignored, and the new Layer 6 readiness gate remained distinct from the existing cortex-index and cortex-embeddings gates.
- [DONE] Phase 3 added red contracts for `prewarm-dense.mjs`, `dense-readiness.mjs`, `dense-readiness.gate.mjs`, and the MCP default-on plus degradation behavior; the red evidence also captured the repo-wide Jest CLI shift to `--testPathPatterns`.
- [DONE] Phase 4 delivered `scripts/semantic-index/prewarm-dense.mjs`, `scripts/semantic-index/dense-readiness.mjs`, `scripts/agent-customization/gates/dense-readiness.gate.mjs`, the `validate-embeddings.mjs` reuse needed by the readiness probe, the MCP `search_corpus` readiness cache plus degradation metadata, the `use_dense.default = true` schema update, and the `package.json` prewarm/readiness scripts.
- [DONE] Phase 5 proved the full end-to-end contract on the real corpus: prewarm passed, warm readiness and gate checks passed, omitted `use_dense` returned `dense_state: "warm"`, forced-cold degradation returned `dense_degraded: true` with a non-empty `dense_reason`, and the focused Jest slices stayed green.
- [DONE] Phase 6 documented the bootstrap contract in `scripts/semantic-index/README.md`, added the post-clone bootstrap command to `CLAUDE.md`, and refreshed the MCP schema descriptions so operators can see the default-on dense recovery path.
- [DONE] Phase 7 compressed and archived this tracker, added the same-boundary audit log, refreshed the plans index and roadmap to point at the archive, and left the completed archive pair as the only Layer 6 tracker source of truth.

## Audit summary

- `node scripts/semantic-index/prewarm-dense.mjs --json` passed on the real corpus, with the model-download step skipped when the cached model was already present and the embed plus validate steps succeeding.
- `node scripts/semantic-index/dense-readiness.mjs --json` and `node scripts/agent-customization/gates/dense-readiness.gate.mjs --json` proved warm dense readiness on the rebuilt post-archive corpus.
- The MCP runtime contract was proven both ways: omitted `use_dense` returned `dense_state: "warm"` without `dense_degraded`, and `DENSE_FORCE_STATE=cold` returned `dense_degraded: true`, `dense_state: "cold"`, and a non-empty `dense_reason`.
- Focused validation covered the new dense-prewarm test boundaries plus the repaired MCP plan-parser regression boundary.
- The final archive validation is recorded in the matching audit log, including the post-archive corpus rebuild, prewarm refresh, readiness pass, and tracker-closure gates.

## Reopen conditions

Reopen this layer only for changes to the dense bootstrap contract, readiness-state semantics, MCP default-on dense provenance fields, readiness-gate behavior, or the operational bootstrap docs.

Do not reopen this plan for Layer 5 embedding-quality work, NeatChat-local retrieval, or broader Repo Cortex indexing behavior unless the change directly alters the Layer 6 default-on dense contract.

## Audit log

See `plans/completed/Semantic_Knowledge_Dense_Prewarm.logs.md`.