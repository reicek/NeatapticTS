---
name: research-methodology
description: 'Use when: executing disciplined Cortex-first discovery workflows.'
argument-hint: 'Describe the investigation target, suspected subsystem, whether Cortex search is needed, known plan files, and whether the goal is reconnaissance only or an implementation brief.'
user-invocable: false
disable-model-invocation: false
skills:
  - plan-alignment
  - repo-cortex-workflow
  - plan-sync-validation
tools:
  - neataptic-cortex-mcp-search_corpus
  - neataptic-cortex-mcp-search_advanced
  - neataptic-cortex-mcp-search_context
  - neataptic-cortex-mcp-load_document
  - neataptic-cortex-mcp-load_chunk
  - neataptic-cortex-mcp-freshness_check
  - neataptic-cortex-mcp-traverse_graph
  - neataptic-cortex-mcp-expand_query
  - neataptic-cortex-mcp-parallel_search
  - neataptic-cortex-mcp-multi_hop_search
  - neataptic-workflow-mcp-get_active_workflow_snapshot
model: anthropic/claude-sonnet-4-20250514
compatibility: 'Works with all NeatapticTS agents and skills that need codebase reconnaissance.'
---

# Research Methodology Playbook

Use this skill when a task requires structured investigation before
implementation, testing, or documentation work begins.

This skill owns the durable discovery-order policies, Cortex-first search
standards, certainty-threshold gating, and context-window mitigation patterns
for NeatapticTS. It ensures that reconnaissance work is systematic, reproducible,
and aligned with the repo's plan-aware execution model.

When a task touches architecture, roadmap items, major refactors, or new
subsystems, this skill coordinates with `plan-alignment` to preserve roadmap
terminology and constraints.

## Core Promise

`research-methodology` exists to make investigation work disciplined and
efficient.

The default promise is:

- Cortex-first search with dense reranking and RRF fusion for semantic queries,
- Ordered README reconnaissance before deep code search,
- Plan-aware execution that notes which documents informed the change,
- Certainty-threshold gating that stops investigation below 90%,
- Context-window mitigation that updates plans and provides handoff prompts.

## When to Use

- A task requires investigation before implementation can begin safely.
- The right plan document or README surface is not obvious yet.
- A semantic search across the corpus would find relevant patterns faster than
  manual grep or glob.
- The certainty level for requirements or environment is below 95%.
- Context window is insufficient and a handoff to a companion agent is needed.
- A demo symptom may indicate a library-level gap that needs investigation.

## When NOT to use

Do NOT use for implementation work - use `04-implementing` instead. Do NOT use for plan creation - use `01-planning` instead.

## Workflow Diagram

```text
Flowchart summary: "Investigation target" → "Check Cortex freshness"; "Check Cortex freshness" → "Index fresh?"; "Index fresh?" → "Search corpus" (Yes), "Fall back to grep/glob" (No); "Search corpus" → "Search context assembly"; "Fall back to grep/glob" → "Read files directly"; "Search context assembly" → "Load chunk by ID"; "Read files directly" → "Report findings"; "Load chunk by ID" → "Traverse dependency graph"; "Report findings"; "Traverse dependency graph" → "Enough certainty?"; "Enough certainty?" → "Report findings" (Yes), "Expand query" (No); "Expand query" → "Search corpus".
```

## Discovery Order

Always follow this ordered reconnaissance pattern before deep code search:

1. **Nearest folder README.md** — Read the generated overview for the target
   boundary. Treat it as a compressed map of the module's public surface.
2. **Parent folder README.md** — When the task spans sibling areas, read the
   parent README to understand the broader subsystem context.
3. **plans/README.md → plans/Roadmap.md** — For roadmap alignment, read the
   plan trigger index first, then the sequencing authority.
4. **Specific source files** — Only after the above surfaces are understood,
   read the orchestration files and helpers that own the behavior.

This order prevents premature deep dives into code before the architectural
context is clear.

## MCP Tool Name Format

MCP tools are exposed using HYPHENS as separators between the server key
and the tool name. The format is: `<server-key>-<tool-name>`. **NEVER use
underscores in MCP tool names.** For example, call
`neataptic-cortex-mcp-search_corpus` (hyphens), NOT
`neataptic_cortex_mcp_search_corpus` (all underscores). The separator
between server key and tool name is always a HYPHEN.

## Cortex-First Search

This skill is the canonical owner of the Cortex-First Search Policy for
NeatapticTS. All investigation, discovery, research, and file-reading work
should follow this policy before falling back to native tools (`grep`, `glob`,
`view`).

### Slice-aware context retrieval

When a step packet declares a `pre_execute_hook` (for example,
`neataptic-workflow-mcp/get_slice_context` with `{ slice_id: "..." }`), invoke it
**before** any search or file read and use the returned slice context as the
primary source for the active plan, phase step contract, and relevant source
files. Only fall back to direct `read_file`/`view` calls for plan/research files
when the hook is unavailable or Cortex is degraded; treat native file reads as a
**degraded-Cortex fallback only**, not the primary path.

### Turso-Native Search Architecture

The Cortex RAG system is backed by a Turso (libSQL) database accessed through
the async `@libsql/client` driver. The search pipeline uses Turso-native
primitives rather than client-side computation:

- **Native vector search** — embeddings are stored as `F8_BLOB` 8-bit
  quantized vectors and ranked server-side via `vector_top_k` with a DiskANN
  approximate-nearest-neighbor index (`libsql_vector_idx`). A brute-force
  fallback is retained for recall validation.
- **Server-side Reciprocal Rank Fusion (RRF, k=60)** — hybrid BM25 + dense
  ranking is fused server-side using the RRF formula `1/(k+rank)`; the legacy
  alpha-blend approach has been removed.
- **Parallel multi-query retrieval** — `parallel_search` runs multiple SQL
  queries concurrently via `Promise.all` with a `TURSO_CONCURRENCY` limiter
  (default 20) and merges results via RRF with graceful degradation.
- **Server-side context assembly** — `search_context` assembles
  token-budgeted context windows via SQL JOINs rather than client-side
  stitching.
- **Server-side query expansion** — `expand_query` uses ANN-first synonym
  discovery against the indexed embedding space.
- **SQL time-decay** — feedback boosts apply `POWER(0.95, days)` decay
  server-side.

### Cortex MCP Tool Reference

| Tool               | Purpose                                                        |
| ------------------ | -------------------------------------------------------------- |
| `freshness_check`  | Verify index currency before searching.                        |
| `search_corpus`    | BM25 + dense hybrid search over indexed chunks.                |
| `search_advanced`  | Full pipeline: classify, expand, retrieve, re-rank, assemble.  |
| `search_context`   | Token-budgeted context window assembly from retrieval results. |
| `load_chunk`       | Load full chunk content by numeric ID.                         |
| `load_document`    | Load all ordered chunks for a repository path.                 |
| `traverse_graph`   | Entity/relationship graph traversal from seed entities.        |
| `expand_query`     | Domain-aware query expansion with synonym discovery.           |
| `parallel_search`  | Run multiple SQL queries concurrently and merge via RRF.       |
| `multi_hop_search` | Vector → graph → vector multi-hop composition (1–3 hops).      |

### 9-Step Search Order

Before manual file reads, follow this ordered search workflow:

1. Check `neataptic-cortex-mcp:freshness_check` for index currency.
2. Use `neataptic-cortex-mcp:search_corpus` for broad BM25 + dense hybrid
   discovery.
3. Use `neataptic-cortex-mcp:search_advanced` with `compact: true` for
   agent-facing queries (includes reranking, ranking explanations,
   `read_top_result`, `follow_up_refs`).
4. Use `neataptic-cortex-mcp:search_context` for token-budgeted context window
   assembly.
5. Use `neataptic-cortex-mcp:load_chunk` to read full chunk content by ID.
6. Use `neataptic-cortex-mcp:load_document` to load all chunks for a file path.
7. Use `neataptic-cortex-mcp:traverse_graph` for entity/dependency graph
   traversal.
8. Use `neataptic-cortex-mcp:expand_query` for domain-aware query expansion.
9. Use `neataptic-cortex-mcp:parallel_search` for concurrent multi-query
   retrieval (merged via RRF), or `neataptic-cortex-mcp:multi_hop_search` for
   vector → graph → vector composed discovery.
10. Fall back to native tools (`grep`, `glob`, `view`) ONLY when Cortex is
    degraded, the target is a known file path, or Cortex returned zero results.

### Gap Escalation Rule

If Cortex RAG cannot answer a needed query, report the gap and suggest an RAG
enhancement. Use native tools as a temporary fallback only.

### Required Workflow

1. Run `npm run index:prewarm` when `dense_state` reports cold or model-only.
2. Use `load_document` for full-file context when a chunk is insufficient.
3. Use `load_chunk` for targeted retrieval when the chunk ID is known.
4. Use `freshness_check` to validate index currency before relying on results.
5. Use `search_advanced` for agent-facing queries that need reranking or
   context assembly.
6. Use `search_context` when a token-budgeted context window is needed.
7. Use `traverse_graph` for dependency and entity graph exploration.
8. Use `expand_query` to broaden a query before retrieval when initial results
   are sparse.
9. Use `parallel_search` to run several SQL queries concurrently and merge
   them via RRF — ideal when multiple distinct query formulations should be
   fused in a single pass.
10. Use `multi_hop_search` for composed vector → graph → vector discovery
    (1–3 hops) when a single retrieval pass cannot bridge the conceptual gap.
11. Treat BM25 + dense reranking fused via RRF as the default search mode.

**Example search queries:**

```text
# Find worker transport patterns
query: "worker payload structured clone transfer list SharedArrayBuffer"
use_dense: true
limit: 5

# Find checkpoint persistence patterns
query: "checkpoint save restore RNG state strict resume"
use_dense: true
limit: 10

# Find NEAT speciation distance calculations
query: "compatibility distance disjoint excess weight difference speciation"
use_dense: true
limit: 5
```

## Plan-Aware Execution

When investigation informs architecture or major refactors:

1. Invoke `plan-alignment` to identify the primary plan document.
2. Note which README and plan document informed the change.
3. Preserve plan terminology and goals unless the user asks to revise them.
4. Call out any visible code/plan mismatch instead of silently drifting.
5. Keep summaries high-level by default; expand only when requested.

**Compact example:**

```text
Use plan-alignment for feed-forward runtime behavior in Flappy Bird.
Trigger phrases: feed-forward builder, runtime contract, demo mismatch.
Core NEAT correctness: maybe adjacent, but not primary.
Goal: identify the primary plan file and any roadmap mismatch risks.
```

## Certainty Thresholds

End every user-facing response with `(Certainty: NN%)`.

| Certainty | Action Required                                           |
| --------- | --------------------------------------------------------- |
| < 90%     | Stop and investigate before proceeding. Do not implement. |
| 90-94%    | Investigate further and ask follow-up questions.          |
| ≥ 95%     | Requirements and environment are clear enough to proceed. |

**Investigation protocol below 90%:**

1. State what is unknown or ambiguous.
2. Name the specific files, plans, or surfaces that need inspection.
3. Recommend a companion agent or skill to investigate the gap.
4. Do not proceed with implementation until certainty reaches 95%.

## Context Window Mitigation

When a change requires more context than is currently available:

1. **Update the source plan document** with a `NEXT:` item:
   ```text
   NEXT: Investigate src/architecture/network/activate/ runtime contract.
   Reason: Current context insufficient to determine slab fast-path boundaries.
   ```
2. **Provide a handoff prompt** in a text-copy box:
   ```text
   Handoff prompt:
   Use research-methodology for network activation runtime contract.
   Known: slab fast-path exists, object traversal fallback present.
   Unknown: exact boundary conditions, performance thresholds.
   Question: What are the slab eligibility rules and when does fallback occur?
   ```
3. **Name the companion agent** that should investigate (e.g., `Boundary Mapper`,
   `Docs Scout`, `Plan Scout`).

## Demo-First Library Gap Policy

When investigation starts from a demo or example symptom:

1. Treat the demo as evidence of a **library DX gap first**.
2. Prefer fixing the library, public API, or shared runtime semantics.
3. Use demo-local compensation only when the issue is genuinely demo-specific.
4. Flag temporary demo-local workarounds as technical debt with a note about
   the preferred library-level fix.

**Investigation pattern:**

```text
Demo symptom: Flappy Bird example requires manual network configuration.
Library gap hypothesis: Architect facade missing sensible defaults for recurrent networks.
Investigation target: src/architecture/architect.ts public API surface.
Preferred fix: Add Architect.recurrent() convenience method.
Demo-local workaround: Accept manual configuration but flag as TODO.
```

## Required Workflow

1. **Identify the investigation target** — subsystem, feature, or symptom.
2. **Run ordered discovery** — README → parent README → plans → source files.
3. **Execute Cortex-first search** when semantic patterns are needed.
4. **Invoke plan-alignment** for architecture or major refactors.
5. **Assess certainty** — gate below 90%, investigate below 95%.
6. **Mitigate context limits** — update plans, provide handoff prompts.
7. **Report findings** — primary plan used, mismatch risks, safest next step.

## Companion Agent Contract

If a companion agent uses this skill, it should:

1. Name this skill explicitly as `research-methodology`.
2. Pass concrete reconnaissance findings into the skill instead of paraphrasing.
3. Keep the agent prompt focused on read-only discovery when the agent is a
   scout.
4. Avoid restating the full workflow or guardrails that already live here.
5. Recommend `plan-alignment` explicitly when architecture or roadmap alignment
   is needed.

## Concrete Search Examples

**Finding a specific function:**

```text
Query: "buildGRU config validation"
Class: code_specific
Approach: search_corpus with use_dense=true, use_rerank=true
```

**Cross-module investigation:**

```text
Query: "how does checkpointing interact with multithread evaluation"
Class: cross_boundary
Approach: search_context with budget=2048, expand_query=true
```

**Exploratory research:**

```text
Query: "what patterns exist for worker payload serialization"
Class: exploratory
Approach: search_advanced with expand_query=true, read_top_result=true
```

## Guardrails

- Do not proceed with implementation when certainty is below 95%.
- Do not skip README reconnaissance and jump directly to source files.
- Do not use Cortex search without prewarming when `dense_state` is cold.
- Do not edit generated `src/**/README.md` files directly; improve JSDoc and
  regenerate.
- Do not use plan-language (roadmap steps, phase names, PR numbers) in
  public-facing documentation; those belong in `plans/` only.
- Do not default to demo-local compensation when investigation points to a
  reusable library fix.

## Cortex Freshness Hooks (Phase 3)

Phase 3 introduced two automated Cortex hooks that maintain index freshness
without agent intervention. Researchers and implementers should be aware of
these hooks but do NOT need to trigger them manually.

### Post-Write Reindex Hook

- **Trigger:** fires after every `edit`/`create`/`apply_patch` tool call.
- **Action:** extracts the written file path and spawns a fire-and-forget
  background process that calls `targeted-reindex.mjs → reindexFiles([path])`
  to re-index only the touched file.
- **Eligibility:** `.md`/`.ts`/`.mjs`/`.js` files under `plans/`,
  `.github/skills/`, `.github/agents/`, `src/`, `examples/`,
  `scripts/agent-customization/`, `rag-index/`, `scripts/mcp-semantic/`.
- **No-op for ineligible files.** Never blocks the host tool.
- Logs to `artifacts/post-write-reindex.log`.

### Pre-Dispatch Freshness Hook

- **Trigger:** fires at SessionStart and before each dispatch.
- **Action:** checks index age against a grace window (default 300s,
  `CORTEX_GRACE_WINDOW_S`) and staleness threshold (default 300s,
  `CORTEX_STALENESS_THRESHOLD_S`). If the index is stale beyond grace +
  threshold, triggers a background full reindex.
- **Never blocks dispatch.** Tooling errors degrade gracefully.
- For complex slices, the orchestrator MAY set `wait_for_reindex: true`.

### Prevention-Over-Detection Policy

The Phase 3 hooks embody a **prevention-over-detection** policy: rather than
waiting for stale-search symptoms to surface (wrong results, missing
chunks, coverage gaps), the system proactively reindexes files immediately
after writes and refreshes the full index when staleness exceeds the
configured threshold. This means:

- Researchers should NOT manually run `build-index.mjs` or `embed-index.mjs`
  after editing corpus files — the post-write hook handles it.
- If search results seem stale, check `artifacts/post-write-reindex.log` and
  `artifacts/pre-dispatch-freshness.log` to confirm the hooks are running,
  rather than re-running the index manually.
- The freshness manifest (`rag-index/data/freshness-manifest.json`) tracks
  per-family freshness, the last reindex timestamp, and can be inspected to
  diagnose staleness.

## Expected Final Output

A strong research-methodology pass should report:

- the investigation target and discovery order followed,
- Cortex search queries used and key results found,
- the primary plan document consulted and why,
- certainty level with justification,
- any context-window mitigation actions taken (plan updates, handoff prompts),
- whether the demo-first library gap policy applied,
- the safest aligned next step for implementation or further investigation.
