# Semantic Knowledge Foundation Log

**Status:** [DONE]

## Completion record

- [DONE] Repo Cortex Layer 1 delivered a build-time SQLite corpus index with document freshness proofs, overlapping markdown chunks, FTS5/BM25 search, JSON-capable CLIs, validation gating, npm scripts, and ignored generated DB output.
- [DONE] Script documentation is available in `scripts/semantic-index/README.md`; generated `data/semantic-index.sqlite` remains local and ignored.
- [DONE] Validation evidence recorded in the closed tracker covers focused Jest, real corpus build, BM25 query over three families, index validation, unchanged-build freshness skip, help output, and final index refresh.
- [DONE] Downstream MCP tools baseline is now archived at `plans/completed/Semantic_Knowledge_MCP_Tools.plans.md`; Layer 2 used `node scripts/semantic-index/validate-index.mjs --json` as the foundation gate.
- [DONE] Closure discrepancy repaired: removed the stale active duplicate `plans/Semantic_Knowledge_Foundation.plans.md` after confirming the compressed archive and matching log already existed under `plans/completed/`.

## Residual risks

- `data/semantic-index.sqlite` is a generated local artifact; downstream MCP work must keep a missing or stale DB failure path with a clear fix hint.
- Corpus freshness depends on rerunning `node scripts/semantic-index/build-index.mjs` after plan or generated-doc edits because `plans/**/*.md` and `src/**/README.md` are indexed sources.
- The current foundation is BM25-only by design; dense retrieval belongs to the later embeddings plan.
