# Semantic Knowledge Foundation (Repo Cortex - Layer 1)

**Status:** [DONE]

## Scope

Closed foundation layer for the Repo Cortex semantic helping lane. This workstream built the offline SQLite-backed corpus index that scans generated READMEs, skills, agents, plans, completed baselines, demo summaries, benchmark headers, root docs, and Copilot instructions into BM25-searchable chunks with SHA-256 freshness proofs.

Non-goals remain out of scope: MCP tool exposure, browser snapshots, dense embeddings, NeatChat conversational memory, and `src/` library changes.

## Final state

- [DONE] Planning and roadmap sync confirmed this layer as the required first dependency for Repo Cortex work.
- [DONE] Research confirmed no prior `scripts/semantic-index/` implementation, identified `fast-glob`, and selected `better-sqlite3` as the required SQLite driver.
- [DONE] Red contracts were added for freshness proofs, markdown chunking, and validation behavior under the focused semantic-index Jest slice.
- [DONE] Implementation added `scripts/semantic-index/schema.sql`, `init-schema.mjs`, `freshness.mjs`, `chunker.mjs`, `build-index.mjs`, `query-index.mjs`, `validate-index.mjs`, npm aliases, and gitignore coverage for `data/semantic-index.sqlite`.
- [DONE] Green validation proved the real build/query/validate path, freshness skip behavior, and the focused Jest command.
- [DONE] Documentation added durable CLI help and `scripts/semantic-index/README.md` for schema, corpus families, freshness, rebuild, query, and validation usage.
- [DONE] Final logging archived this compressed tracker and recorded the same-boundary completion log.

## Audit summary

Key validation evidence captured during the workstream:

- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/semantic-index` passed 1 suite / 3 tests.
- `node scripts/semantic-index/build-index.mjs` built the corpus and a second unchanged run skipped all indexed documents.
- `node scripts/semantic-index/query-index.mjs "NEAT activation" --json` returned ranked results across `demo`, `plan`, and `readme` families.
- `node scripts/semantic-index/validate-index.mjs --json` passed with 830 documents and at least 27,726 chunks before final archive refresh.
- `node scripts/semantic-index/build-index.mjs --json; node scripts/semantic-index/validate-index.mjs --json` passed immediately before Step 07 started.

Layer 2 closeout: `plans/completed/Semantic_Knowledge_MCP_Tools.plans.md` is archived as [DONE]. Its implementation used the archived foundation baseline plus the live `node scripts/semantic-index/validate-index.mjs --json` freshness gate.

## Reopen conditions

Reopen this archive only if the offline corpus index contract changes materially, such as adding new corpus families, changing schema or FTS behavior, replacing SQLite, changing freshness proof semantics, or making the index a CI-mandatory gate.

Routine follow-up should start from active downstream plans instead:

- `plans/completed/Semantic_Knowledge_MCP_Tools.plans.md` for the archived MCP tool exposure baseline.
- `plans/Semantic_Knowledge_Browser_Snapshot.plans.md` for browser JSON snapshots and IndexedDB loading.
- `plans/Semantic_Knowledge_Embeddings.plans.md` for hybrid BM25 plus dense retrieval.

## Audit log

See `plans/completed/Semantic_Knowledge_Foundation.logs.md`.
