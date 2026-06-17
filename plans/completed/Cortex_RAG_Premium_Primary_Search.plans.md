# Cortex RAG Premium Primary Search

**Status:** [DONE]

Claim: 06-documenting

## Purpose and background

The Repo Cortex advanced RAG architecture ([completed/Repo_Cortex_Advanced_RAG_Architecture.plans.md](completed/Repo_Cortex_Advanced_RAG_Architecture.plans.md)) is implemented and archived. This follow-up workstream made the Cortex the premium primary search for LLM agents in this repository: always fresh, concise by default, high-signal, and usable in a single call.

It addressed three concerns raised by agents using the earlier tools:

1. **Freshness hooks** — smart post-write hooks keep the index current without blocking the agent.
2. **Quality gaps** — tokenization, ranking, graph traversal, symbol lookup, determinism, and feedback rough edges were closed.
3. **LLM ergonomics** — default result sets were compacted, generated READMEs were deprioritized for code queries, and follow-up reads became single-call.

Non-goals:

- Do not replace the existing BM25/dense/hybrid architecture; extend it.
- Do not add cloud LLM dependencies; stay local-first.
- Do not change `src/` library behavior.

## Scope

- Smart freshness hooks (debounced, batched, non-blocking).
- Premium primary search gap closure (tokenization, boosting, reranking, graph, symbol lookup, latency, determinism, README filtering, fallback, transparency, feedback).
- LLM-useful defaults and options (compact size, compact mode, single-call search-and-read, follow-up refs, ranking explanation, code-only filter, auto fallback).
- Quality-over-quantity result design (tiered results, deduplication, budget tracking).
- MCP tool option updates, evaluation, and documentation.

## Trigger phrases

- premium primary search, primary search, cortex premium search, single-call search, search-and-read, compact search results
- smart freshness hooks, post-write index update, debounced index update, incremental index update, freshness hooks
- BM25 tokenization, code identifier tokenization, code-source boost, ts-source boost
- reranker effectiveness, cross-encoder rerank, graph traversal serialization, exact symbol lookup
- cold-start embeddings, embedding determinism, ANN determinism, README noise filter
- native search fallback, auto fallback, follow-up refs, explain ranking, include_code_only
- result tiers, essential results, supporting results, supplementary results, search budget

## Implementation phases

### Phase 1 — Planning and gap consolidation [DONE]

> **Coverage note:** Consolidated premium-primary-search gaps from the completed advanced RAG plan, authored Step 02–12 packets with slices, defined MCP option names and defaults, updated `plans/README.md` and `plans/Roadmap.md`, and passed plan-phase-step and plan-sync validation.

```yaml
phase: 1
title: 'Planning and gap consolidation'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_phase: 'Phase 2 — Smart freshness hooks'
skills:
  - 'plan-alignment'
  - 'phase-handoff-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
acceptance_criteria:
  - 'All phases and steps conform to the step-packet schema.'
  - 'README and Roadmap reference the plan with aligned status.'
placeholder_steps:
  - 'Step 01 — Consolidate gaps and author step packets'
```

#### Step 01 — Consolidate gaps and author step packets [DONE]

> **Coverage note:** Step packets authored and validated; README/Roadmap references added.

```yaml
phase: 1
step: 1
title: 'Consolidate gaps and author step packets'
status: '[DONE]'
goal: 'planning'
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_step: null
skills:
  - 'plan-alignment'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
```

---

### Phase 2 — Smart freshness hooks [DONE]

> **Coverage note:** Added a non-blocking post-write hook that keeps the Repo Cortex index incrementally fresh. BM25 term updates happen immediately; embedding updates are queued and applied incrementally. A freshness proof is auto-touched on every successful update, and failures degrade gracefully without blocking the write.

```yaml
phase: 2
title: 'Smart freshness hooks'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_phase: 'Phase 3 — Premium primary search quality gaps'
skills:
  - 'plan-alignment'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
acceptance_criteria:
  - 'Freshness hooks update BM25 immediately and embeddings incrementally.'
  - 'Failures degrade gracefully without blocking the triggering write.'
placeholder_steps:
  - 'Step 02 — Implement post-write freshness hooks'
```

#### Step 02 — Implement post-write freshness hooks [DONE]

> **Coverage note:** Post-write freshness hooks implemented and validated.

```yaml
phase: 2
step: 2
title: 'Implement post-write freshness hooks'
status: '[DONE]'
goal: 'implementing'
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_step: null
skills:
  - 'implementation-standards'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
```

---

### Phase 3 — Premium primary search quality gaps [DONE]

> **Coverage note:** Closed the quality gaps that reduced trust in primary search: identifier-aware BM25 tokenization and `ts-source` boosting for code queries; reranker effectiveness and graph-traversal serialization hardening; exact symbol lookup via `symbol_name` index and cold-start latency caching; determinism, README noise suppression, and a native-search fallback router; freshness transparency and feedback normalization.

```yaml
phase: 3
title: 'Premium primary search quality gaps'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_phase: 'Phase 4 — LLM-useful defaults and options'
skills:
  - 'plan-alignment'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
acceptance_criteria:
  - 'Tokenization, ranking, graph traversal, symbol lookup, determinism, and feedback are hardened.'
  - 'Quality fixes integrate cleanly with the existing hybrid pipeline.'
placeholder_steps:
  - 'Step 03 — Harden BM25 tokenization and code-source boosting'
  - 'Step 04 — Fix reranker effectiveness and graph traversal serialization'
  - 'Step 05 — Add exact symbol lookup and cold-start latency improvements'
  - 'Step 06 — Harden determinism, suppress README noise, add fallback router'
  - 'Step 07 — Add freshness transparency and feedback normalization'
```

#### Step 03 — Harden BM25 tokenization and code-source boosting [DONE]

> **Coverage note:** Identifier-aware tokenization shipped (`camelCase`, `snake_case`, dotted/file-extension identifiers preserved); `code_specific` queries route to `ts-source` family.

```yaml
phase: 3
step: 3
title: 'Harden BM25 tokenization and code-source boosting'
status: '[DONE]'
goal: 'implementing'
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_step: 'Step 04 — Fix reranker effectiveness and graph traversal serialization'
skills:
  - 'implementation-standards'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
```

#### Step 04 — Fix reranker effectiveness and graph traversal serialization [DONE]

> **Coverage note:** Reranker effectiveness and graph traversal serialization hardened.

```yaml
phase: 3
step: 4
title: 'Fix reranker effectiveness and graph traversal serialization'
status: '[DONE]'
goal: 'implementing'
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_step: 'Step 05 — Add exact symbol lookup and cold-start latency improvements'
skills:
  - 'implementation-standards'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
```

#### Step 05 — Add exact symbol lookup and cold-start latency improvements [DONE]

> **Coverage note:** Exact symbol lookup uses `symbol_name` index; cold-start latency cached via `getEmbedReadiness`.

```yaml
phase: 3
step: 5
title: 'Add exact symbol lookup and cold-start latency improvements'
status: '[DONE]'
goal: 'implementing'
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_step: 'Step 06 — Harden determinism, suppress README noise, add fallback router'
skills:
  - 'implementation-standards'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
```

#### Step 06 — Harden determinism, suppress README noise, add fallback router [DONE]

> **Coverage note:** Determinism hardened, README noise suppressed for code queries, native-search fallback router added to `search_advanced`.

```yaml
phase: 3
step: 6
title: 'Harden determinism, suppress README noise, add fallback router'
status: '[DONE]'
goal: 'implementing'
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_step: 'Step 07 — Add freshness transparency and feedback normalization'
skills:
  - 'implementation-standards'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
```

#### Step 07 — Add freshness transparency and feedback normalization [DONE]

> **Coverage note:** Freshness proof now carries `mtime_ms`, `size`, and `sha256` from the most recently indexed document; feedback normalization aligned boosts with the corpus graph.

```yaml
phase: 3
step: 7
title: 'Add freshness transparency and feedback normalization'
status: '[DONE]'
goal: 'implementing'
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_step: null
skills:
  - 'implementation-standards'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
```

---

### Phase 4 — LLM-useful defaults and options [DONE]

> **Coverage note:** Shipped the LLM-ergonomic defaults: conservative response sizes and `compact` mode; single-call `search_context` with `read_top_result` and `follow_up_refs`; `explain_ranking`, `include_code_only`, and `auto_fallback` in `search_advanced`.

```yaml
phase: 4
title: 'LLM-useful defaults and options'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_phase: 'Phase 5 — Integration, evaluation, and documentation'
skills:
  - 'plan-alignment'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
acceptance_criteria:
  - 'Compact mode, single-call search-and-read, follow-up refs, ranking explanation, and code-only filter are wired end-to-end.'
  - 'Defaults are applied at the MCP handler level without breaking explicit overrides.'
placeholder_steps:
  - 'Step 08 — Implement conservative default response sizes and compact mode'
  - 'Step 09 — Implement single-call search-and-read and follow-up refs'
  - 'Step 10 — Implement explain_ranking, include_code_only, and auto_fallback'
```

#### Step 08 — Implement conservative default response sizes and compact mode [DONE]

> **Coverage note:** `compact: true` default applied to `search_corpus` and `search_context`; result sets trimmed to high-signal fields.

```yaml
phase: 4
step: 8
title: 'Implement conservative default response sizes and compact mode'
status: '[DONE]'
goal: 'implementing'
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_step: 'Step 09 — Implement single-call search-and-read and follow-up refs'
skills:
  - 'implementation-standards'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
```

#### Step 09 — Implement single-call search-and-read and follow-up refs [DONE]

> **Coverage note:** `search_context` defaults `read_top_result: true` and returns `follow_up_refs` graph traversals in one call.

```yaml
phase: 4
step: 9
title: 'Implement single-call search-and-read and follow-up refs'
status: '[DONE]'
goal: 'implementing'
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_step: 'Step 10 — Implement explain_ranking, include_code_only, and auto_fallback'
skills:
  - 'implementation-standards'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
```

#### Step 10 — Implement explain_ranking, include_code_only, and auto_fallback [DONE]

> **Coverage note:** `search_advanced` gained `explain_ranking`, `include_code_only` default for `code_specific`, and `auto_fallback: true`.

```yaml
phase: 4
step: 10
title: 'Implement explain_ranking, include_code_only, and auto_fallback'
status: '[DONE]'
goal: 'implementing'
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_step: null
skills:
  - 'implementation-standards'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
```

---

### Phase 5 — Integration, evaluation, and documentation [DONE]

> **Coverage note:** Integrated all premium-search improvements into a coherent pipeline, ran the RAG evaluation suite, updated MCP tool schemas and the `scripts/mcp-semantic/README.md` user-facing documentation, compressed the plan history, and archived the plan/log pair.

```yaml
phase: 5
title: 'Integration, evaluation, and documentation'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_phase: null
skills:
  - 'plan-alignment'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
acceptance_criteria:
  - 'Premium search pipeline integrates hooks, quality fixes, and new defaults.'
  - 'Eval suite reports metrics without errors or timeouts.'
  - 'MCP tool documentation and tracker closure are complete.'
placeholder_steps:
  - 'Step 11 — Integrate premium search pipeline and run eval suite'
  - 'Step 12 — Document new MCP options and close tracker'
```

#### Step 11 — Integrate premium search pipeline and run eval suite [DONE]

> **Coverage note:** Pipeline integration and eval suite complete. `eval-runner` forwards premium defaults and reports `premium_defaults_applied`; MCP handlers apply compact/read_top_result/auto_fallback defaults; all targeted tests, preflight checks, and Tier-1 gates passed.

```yaml
phase: 5
step: 11
title: 'Integrate premium search pipeline and run eval suite'
status: '[DONE]'
goal: 'implementing'
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_step: 'Step 12 — Document new MCP options and close tracker'
skills:
  - 'implementation-standards'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
  - 'node scripts/semantic-index/rag-eval/eval-runner.mjs --json'
```

#### Step 12 — Document new MCP options and close tracker [DONE]

> **Coverage note:** Updated `scripts/mcp-semantic/README.md` with new tool options and defaults for `search_corpus`, `search_context`, and `search_advanced`; regenerated docs with `npm run docs`; compressed the plan; created this matching log; moved the plan/log pair to `plans/completed/`; aligned `plans/README.md` and `plans/Roadmap.md`; closure gates passed.

```yaml
phase: 5
step: 12
title: 'Document new MCP options and close tracker'
status: '[DONE]'
goal: 'documenting'
tdd_sequence: 'green-only'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Cortex_RAG_Premium_Primary_Search.plans.md'
copy_paste: true
next_step: null
skills:
  - 'documenting'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md'
  - 'npm run docs'
acceptance_criteria:
  - 'MCP tool documentation lists all new options and defaults.'
  - 'Plan is compressed and archived with matching log file.'
  - 'Stale-wip-plans and log-completion-marker gates pass.'
```

## Validation gates

Run these gates to confirm the plan and its registration are healthy:

```bash
node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md
node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Cortex_RAG_Premium_Primary_Search.plans.md
```

Run these tracker closure gates after Step 12:

- `node scripts/agent-customization/gates/phase-compression.gate.mjs --json`
- `node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json`
- `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json`

## Latest validation evidence

- Step 12 closure pass: `npm run docs` completed successfully; `validate-plan-phase-packets` and `validate-plan-sync` both reported `0 errors, 0 warnings` for the active plan; `phase-compression`, `log-completion-marker`, and `stale-wip-plans` closure gates returned `pass: true`; plan/log pair archived under `plans/completed/`; `plans/README.md` and `plans/Roadmap.md` aligned to `[DONE]`.
