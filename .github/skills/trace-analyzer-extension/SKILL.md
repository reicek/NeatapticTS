---
name: trace-analyzer-extension
description: 'Use when: extending the trace analyzer with new rollups or comparisons.'
argument-hint: 'Describe the trace question, missing metric, and validation trace file.'
user-invocable: true
disable-model-invocation: false
skills:
  - trace-audit-reporting
  - performance-optimization
  - tracker-handoff
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Trace Analyzer Extension Playbook

Use this skill when `scripts/analyze-trace/analyze-trace.ts` needs to be
modified — not just consumed — because its current output cannot answer a
concrete engineering question about a Chrome or Perfetto trace.

This skill owns the durable workflow for extending the repo's trace analyzer
tooling. It is the companion to `trace-audit-reporting`: that skill consumes
analyzer output to write reports; this skill extends the analyzer when the
output is insufficient. When extension work also updates a durable tracker or
report log, `tracker-handoff` owns the plan/log structure.

## When to Use

- The existing analyzer output is not enough to explain a performance problem.
- A trace audit needs a new deterministic rollup or comparison section.
- A hotspot needs better attribution by bundle, worker URL, thread, or event
  type.
- The report needs percentiles, top-N comparisons, or a tighter summary format.
- A one-off shell command would be too fragile or too noisy to repeat reliably
  across future trace captures.

## When NOT to use

Do NOT use for trace reporting or analysis - use `trace-audit-reporting` instead. Do NOT use for performance optimization implementation - use `performance-optimization` instead.

## Workflow Diagram

```text
Flowchart summary: "Analyzer cant answer question" → "Read analyze-trace.ts"; "Read analyze-trace.ts" → "Design new section"; "Design new section" → "Implement helper"; "Implement helper" → "Validate against trace"; "Validate against trace" → "Output correct?"; "Output correct?" → "Update trace-audit-reporting docs" (Yes), "Design new section" (No); "Update trace-audit-reporting docs" → "Done"; "Done".
```

## Task Packet

Pass a compact packet that names the engineering question the current analyzer
cannot answer, the missing metric or section, and the trace file to validate
against.

```text
Use trace-analyzer-extension to add per-thread percentile summaries.
Engineering question: what is the p95 duration of HandlePostMessage events on the
  worker thread, compared to the renderer thread?
Missing section: percentile breakdown by thread for selected event types.
Validation trace: traces/flappy-worker-2026-05.json.
Output format: compact table, one row per thread, columns p50/p95/max.
```

## Primary Resources

- [Analyzer extension workflow](./references/analyzer-extension-workflow.md)
- [Extension checklist](./assets/extension-checklist.md)

## Required Workflow

1. State the engineering question the current analyzer cannot answer. Make it
   concrete: "What is X for Y events on Z thread?"
2. Read `scripts/analyze-trace/analyze-trace.ts` in full before proposing
   any changes. Understand the current section structure, helper functions,
   and CLI behavior.
3. Prefer extending existing helpers over adding parallel ad hoc logic. Reuse
   the existing thread-awareness model so renderer, worker, browser, and GPU
   work remain separable in all output.
4. Design the new section output to be:
   - deterministic (same trace → same output every run),
   - text-first and easy to compare across captures,
   - compact: tables or one-line summaries over verbose prose,
   - ordered by value then stable by name when ordering could vary.
5. Implement the new section or helper in the analyzer script.
6. Validate the new output against a real trace file from the repo:
   ```bash
   npx ts-node scripts/analyze-trace/analyze-trace.ts <trace-path> --top=15
   ```
   Confirm the new section appears and answers the engineering question.
7. If the new section changes how agents should interpret results, update the
   `trace-audit-reporting` skill or its reference docs to reflect the new
   output shape.
8. Keep JSDoc current for any new helpers or changed CLI flags.

## Output Design Rules

- New sections must answer a concrete engineering question.
- Prefer compact tables or one-line summaries over verbose prose.
- Keep ordering deterministic by value, then stable by name when needed.
- Preserve thread awareness so renderer, worker, browser, and GPU work remain
  separable.
- Avoid sections that require manual post-processing to become useful.

## Before/After Analyzer Output

**Before (insufficient):**

```text
Top events by duration:
  HandlePostMessage: 45ms (12 occurrences)
```

**After (with per-thread breakdown):**

```text
Per-thread p95 durations:
| Thread   | p50   | p95   | Max   |
|----------|-------|-------|-------|
| renderer | 2ms   | 8ms   | 15ms  |
| worker   | 5ms   | 22ms  | 45ms  |
```

## Decision Tree

```text
Flowchart summary: "Trace work" → "What kind?"; "What kind?" → "trace-analyzer-extension" (Analyzer cannot answer question), "trace-audit-reporting" (Write report from analyzer output), "performance-optimization" (Implement perf optimization); "trace-analyzer-extension"; "trace-audit-reporting"; "performance-optimization".
```

## Guardrails

- Do not add output that duplicates an existing section with minor formatting
  changes.
- Do not overfit the analyzer to one trace unless the repo clearly needs that
  exact workflow for recurring audits.
- Do not add specific calendar dates to generated report sections, example log
  headings, or companion notes. Keep report titles stable and content-focused.
- Do not add non-deterministic timestamps, colors, or formatting noise.
- Do not pull in a new dependency unless the existing TypeScript runtime cannot
  reasonably support the change.
- Do not replace the current lightweight architecture with a large parser
  abstraction unless the script has clearly outgrown the current shape.
- When extension work updates a durable tracker or report log, follow
  `tracker-handoff` for `[PLANNED]`, `[WIP]`, `[DONE]`, compression, and
  `Handoff query` structure.

## Expected Final Output

A strong trace-analyzer-extension pass should report:

- the engineering question that motivated the change,
- which section or helper was added or modified in
  `scripts/analyze-trace/analyze-trace.ts`,
- the new output shape (table, line format, or summary structure),
- the validation trace used and confirmation that the new section appears
  correctly,
- whether `trace-audit-reporting` reference docs were updated to reflect the
  new output,
- any follow-up extension opportunities identified during the pass.

## Repo-Specific Notes

- The analyzer is intentionally lightweight and runs through `ts-node`.
- The script already supports thread summaries, longest events, event rollups,
  function-call attribution, and percentile summaries for selected events.
- Extend the current architecture instead of replacing it with a large parser
  abstraction unless the script has clearly outgrown the current shape.
