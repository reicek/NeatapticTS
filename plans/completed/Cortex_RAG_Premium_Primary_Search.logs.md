# Cortex RAG Premium Primary Search Log

**Status:** [DONE]

## Audit scope

- Objective: close the Repo Cortex premium primary search workstream after making the corpus index the default, high-signal search surface for LLM agents.
- Coverage included smart freshness hooks, BM25/code-source quality fixes, exact symbol lookup, deterministic ANN, README noise suppression, native search fallback, single-call search-and-read, follow-up refs, ranking explanations, code-only filtering, automatic fallback, MCP tool documentation, and tracker alignment.

## Durable milestones

### [DONE] Planning and gap consolidation

- Extracted premium-primary-search gaps from the archived advanced RAG plan.
- Authored Step 02–12 packets with option names, defaults, and slice structure.
- Registered the plan in `plans/README.md` and `plans/Roadmap.md` and passed plan-sync validation.

### [DONE] Smart freshness hooks

- Added non-blocking post-write hooks that update BM25 terms immediately and queue embedding updates incrementally.
- Freshness proof is auto-touched on every successful update; failures degrade gracefully without blocking the write.

### [DONE] Quality gap closure

- Identifier-aware BM25 tokenization preserves `camelCase`, `snake_case`, and dotted/file-extension identifiers.
- `code_specific` queries route to the `ts-source` family so source chunks outrank generated READMEs.
- Reranker effectiveness and graph-traversal serialization were hardened.
- Exact symbol lookup uses the `symbol_name` index; cold-start latency is cached via `getEmbedReadiness`.
- Determinism hardened, README noise suppressed, and a native-search fallback router added to `search_advanced`.
- Freshness proof carries `mtime_ms`, `size`, and `sha256` from the most recently indexed document.

### [DONE] LLM-useful defaults and options

- `search_corpus` and `search_context` default to `compact: true`.
- `search_context` defaults to `read_top_result: true` and returns `follow_up_refs` graph traversals in one call.
- `search_advanced` defaults to `auto_fallback: true`, `include_code_only: true` for `code_specific` queries, and supports `explain_ranking: true`.

### [DONE] Integration, evaluation, and documentation

- Integrated all premium-search improvements into a coherent `search_advanced`/`search_context` pipeline.
- Eval runner reports `premium_defaults_applied` and ran 68 queries without errors or timeouts.
- Updated `scripts/mcp-semantic/README.md` with new tool options, defaults, representative examples, and a query-routing/identifier-tokenization explainer.
- Regenerated generated docs with `npm run docs`.
- Compressed plan history, created this log, and archived the plan/log pair under `plans/completed/`.
- Aligned `plans/README.md` and `plans/Roadmap.md` to the terminal `[DONE]` state.

## Controls and evidence

- Plan phase packets: `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md` -> PASS (`0 errors, 1 warning` for zero [WIP] phases on a closed active plan).
- Plan registration sync: `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md` -> PASS (`0 errors, 0 warnings`).
- Docs regeneration: `npm run docs` -> PASS (exit 0).
- Closure gates:
  - `node scripts/agent-customization/gates/phase-compression.gate.mjs --json` -> PASS.
  - `node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json` -> PASS.
  - `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json` -> PASS.
- No `src/` files were changed in this workstream; coverage guard was vacuously satisfied.

## Reopen triggers

- A future documented bridge API changes the `repo-static`, `direct-MCP`, `bridge-required`, or manual-only ownership boundaries used by the Cortex MCP servers.
- A new workstream needs to extend premium primary search beyond the shipped local-first surface (for example cloud LLM rerankers, cross-repo search, or non-NEAT domains).
- The archived defaults drift away from the current implementation (for example `compact`, `read_top_result`, `auto_fallback`, or `include_code_only` semantics change without a new plan).
