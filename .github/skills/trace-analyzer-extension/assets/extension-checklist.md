# Trace Analyzer Extension Checklist

Use this checklist before finalizing changes to `scripts/analyze-trace.ts`.

## Before Editing

- Identify the exact engineering question.
- Confirm the current analyzer cannot already answer it.
- Pick the smallest extension type: rollup, attribution, or comparison.

## During Editing

- Keep the new logic deterministic.
- Reuse existing helper patterns for duration conversion and thread labeling.
- Add or update JSDoc for any new helper or CLI option.
- Keep the report readable in a plain terminal.

## Before Finalizing

- Run the analyzer on a real repo trace.
- Confirm the new section is actionable.
- Confirm the new section does not duplicate older output.
- Update the reporting skill if the default workflow changed.
- Summarize the engineering value of the new section, not just the code change.