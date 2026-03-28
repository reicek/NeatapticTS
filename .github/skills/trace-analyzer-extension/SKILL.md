---
name: trace-analyzer-extension
description: 'Extend scripts/analyze-trace/analyze-trace.ts with new rollups, comparisons, script attribution, percentiles, or deterministic report sections. Use when the existing trace analyzer cannot answer an engineering question about Chrome trace or Perfetto data.'
argument-hint: 'Describe the trace question, missing metric, and validation trace file.'
user-invocable: true
disable-model-invocation: false
---

# Trace Analyzer Extension

Use this skill when an agent needs to modify `scripts/analyze-trace/analyze-trace.ts` instead
of only consuming its current output.

## When to Use

- The existing analyzer output is not enough to explain a performance problem.
- A trace audit needs a new deterministic rollup or comparison section.
- A hotspot needs better attribution by bundle, worker URL, thread, or event.
- The report needs percentiles, top-N comparisons, or a tighter summary format.
- A one-off shell command would be too fragile or too noisy to repeat.

## Primary Resources

- [Analyzer extension workflow](./references/analyzer-extension-workflow.md)
- [Extension checklist](./assets/extension-checklist.md)

## Standard Workflow

1. State the engineering question the current analyzer cannot answer.
2. Read `scripts/analyze-trace/analyze-trace.ts` before proposing a new section.
3. Prefer extending existing helpers over adding parallel ad hoc logic.
4. Keep output deterministic, text-first, and easy to compare across captures.
5. Validate the new output against a real trace file from the repo.
6. If the new section changes how agents should interpret results, update the
   reporting skill or its reference docs too.

## Output Design Rules

- New sections must answer a concrete engineering question.
- Prefer compact tables or one-line summaries over verbose prose.
- Keep ordering deterministic by value, then stable by name when needed.
- Preserve thread awareness so renderer, worker, browser, and GPU work remain
  separable.
- Avoid sections that require manual post-processing to become useful.

## Guardrails

- Do not add output that duplicates an existing section with minor formatting
  changes.
- Do not overfit the analyzer to one trace unless the repo clearly needs that
  exact workflow.
- When extension work also updates a durable tracker or report log, follow
  `tracker-handoff` for `[PLANNED]`, `[WIP]`, `[DONE]`, compression, and
  `Handoff query` structure.
- Do not add specific calendar dates to generated report sections, example log
  headings, or companion notes. Keep report titles stable and content-focused.
- Do not add non-deterministic timestamps, colors, or formatting noise.
- Do not pull in a new dependency unless the existing TypeScript runtime cannot
  reasonably support the change.
- Keep JSDoc current when adding helpers or changing CLI behavior.

## Repo-Specific Notes

- The analyzer is intentionally lightweight and runs through `ts-node`.
- The script already supports thread summaries, longest events, event rollups,
  function-call attribution, and percentile summaries for selected events.
- Extend the current architecture instead of replacing it with a large parser
  abstraction unless the script has clearly outgrown the current shape.