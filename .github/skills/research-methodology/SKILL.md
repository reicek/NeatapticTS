---
name: research-methodology
description: 'Execute disciplined discovery workflows in NeatapticTS using Cortex-first search, ordered README reconnaissance, plan-aware execution, and certainty-threshold gating. Use when a task requires structured investigation before implementation, testing, or documentation.'
argument-hint: 'Describe the investigation target, suspected subsystem, whether Cortex search is needed, known plan files, and whether the goal is reconnaissance only or an implementation brief.'
user-invocable: false
disable-model-invocation: false
skills:
  - plan-alignment
tools:
  - neataptic-cortex-mcp-search_corpus
  - neataptic-cortex-mcp-load_document
  - neataptic-cortex-mcp-load_chunk
  - neataptic-cortex-mcp-freshness_check
  - neataptic-workflow-mcp-get_active_workflow_snapshot
model: anthropic/claude-sonnet-4-20250514
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

- Cortex-first search with dense reranking for semantic queries,
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

## Cortex-First Search

When semantic search would find relevant patterns faster than manual lookup:

```text
Use neataptic-cortex-mcp-search_corpus with:
  query: "<conceptual query describing the pattern>"
  use_dense: true
  limit: 10
```

**Required workflow:**

1. Run `npm run index:prewarm` when `dense_state` reports cold or model-only.
2. Use `load_document` for full-file context when a chunk is insufficient.
3. Use `load_chunk` for targeted retrieval when the chunk ID is known.
4. Use `freshness_check` to validate index currency before relying on results.
5. Treat BM25 + dense reranking as the default search mode.

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

## Expected Final Output

A strong research-methodology pass should report:

- the investigation target and discovery order followed,
- Cortex search queries used and key results found,
- the primary plan document consulted and why,
- certainty level with justification,
- any context-window mitigation actions taken (plan updates, handoff prompts),
- whether the demo-first library gap policy applied,
- the safest aligned next step for implementation or further investigation.
