# Analyzer Extension Workflow

This reference explains how to safely extend `scripts/analyze-trace/analyze-trace.ts` without
turning it into a one-off debugging script.

## Start With the Question

Write the missing question in one sentence.

Good examples:

- Which thread owns most `HandlePostMessage` time?
- Which script URLs dominate `FunctionCall` on dropped-frame traces?
- Are frame spikes coming from a few outliers or a broad high-percentile shift?
- How do two traces compare on dropped frames, worker totals, and hottest
  events?

If the question is vague, the output section will usually be vague too.

## Preferred Extension Types

Choose the smallest useful extension.

### 1. New rollup section

Use when the event exists already and only aggregation is missing.

Examples:

- event totals by thread,
- `HandlePostMessage` by worker URL,
- `RunTask` by thread category.

### 2. Better attribution

Use when a current section is correct but not actionable enough.

Examples:

- resolve script URL from event args,
- surface worker URL alongside generic worker thread labels,
- split browser-main and renderer-main owners more clearly.

### 3. Comparison mode

Use when engineers need baseline-versus-candidate reporting.

Examples:

- compare dropped frames between two traces,
- compare top events and top threads side by side,
- compare p50, p90, p99 frame metrics.

Only add comparison mode if the repo is likely to reuse it.

## Implementation Guidance

Follow this order:

1. Add or adjust the smallest data helper needed.
2. Reuse existing normalization helpers for thread labels and durations.
3. Keep the section printer near the existing report flow in `main()`.
4. Preserve the current compact textual style.
5. Add or update JSDoc for new helpers and CLI flags.

## Validation Workflow

Validate against a real trace from the repository.

Recommended command pattern:

```bash
npm run trace:analyze -- <trace-path> --top=15
```

Validation questions:

- Does the new section answer the intended question directly?
- Is the output deterministic between runs?
- Does the new section avoid duplicating older sections?
- Does the script still work for a normal single-trace audit?

## When to Update the Reporting Skill

Update `.github/skills/trace-audit-reporting/` when the analyzer extension
changes the default audit workflow.

Examples:

- a new high-value summary section should be mentioned in the workflow,
- a new comparison mode becomes standard for regression triage,
- a new CLI flag changes the recommended command.

## Anti-Patterns

- Adding raw JSON dumps.
- Adding wide output that is hard to scan in chat or terminals.
- Mixing data collection, formatting, and CLI parsing into one large helper.
- Replacing deterministic summaries with subjective prose inside the script.
- Adding a feature that only makes sense for one trace file with no reusable
  value.