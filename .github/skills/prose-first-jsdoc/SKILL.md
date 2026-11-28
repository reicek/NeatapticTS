---
name: prose-first-jsdoc
description: 'Use when: authoring or repairing the leading file-level JSDoc block of a public src/ module, or when a generated src/**/README.md file section renders with no prose summary because the block opens with a tag such as @module.'
argument-hint: 'Describe the source file or generated README, the current leading JSDoc block, whether the rendered file summary is empty, and whether this pass is detection-only or an edit.'
user-invocable: false
disable-model-invocation: false
skills:
  - updating-js-docs
  - auditing-js-docs
  - educational-docs
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Prose-First JSDoc

This skill governs the **ordering rule** for the leading file-level JSDoc block of a public module: the block must open with prose, and module-identity tags (`@module`, `@packageDocumentation`) must come last.

This is a separate concern from `updating-js-docs` (which governs JSDoc *content* — examples, citations, Mermaid, `@param`/`@returns` completeness) and from `auditing-js-docs` (which governs the *audit workflow*). Those skills can both pass while the generated README still renders an empty file summary, because the generator stops reading prose at the first tag.

## When to Use

- Authoring a new public module under `src/` and writing its leading JSDoc block.
- A generated `src/**/README.md` shows a file heading (`## <path/to/module.ts>`) followed immediately by a symbol heading, with no prose in between.
- Repairing an existing module whose leading block starts with `@module`, `@packageDocumentation`, `@file`, or `@author`.
- Auditing a source surface for modules whose file summary is silently empty.
- Before adding a new module to a subtree (for example `src/observability/telemetry/telemetry.policy.ts`), so the summary renders on the first `npm run docs` run.

## When NOT to use

Do NOT use for exported-symbol JSDoc content — use `updating-js-docs`. Do NOT use for a read-only documentation gap list — use `auditing-js-docs`. Do NOT use for README tone, diagrams, or narrative depth — use `educational-docs`. Do NOT use to restyle generated `src/**/README.md` files; the README is a rendered view of the source JSDoc and is never hand-edited.

## Workflow Diagram

```text
Flowchart summary: "Leading JSDoc block" → "Opens with prose?"; "Opens with prose?" → "Summary renders" (Yes), "Summary is empty" (No); "Summary renders" → "Tag order OK?"; "Summary is empty" → "Move tag to end"; "Tag order OK?" → "Prose-first compliant" (Yes), "Reorder remaining tags" (No); "Move tag to end" → "Summary renders"; "Reorder remaining tags" → "Prose-first compliant"; "Prose-first compliant" → "Regenerate README".
```

## Task Packet

Include the source file path (or the generated README path), the current leading JSDoc block, whether the rendered file summary is empty, and whether the pass is detection-only or an edit.

```text
Use prose-first-jsdoc for src/observability/telemetry/telemetry.policy.ts.
Current leading block: <first tag, or "opens with prose">
Rendered summary in src/observability/telemetry/README.md: <present | empty>
Pass type: <detection-only | edit>
Regenerate with npm run docs after: <yes | no>
```

## Required Workflow

1. Read the top of the target source file and identify the **first** `/** ... */` block. Only the first block is treated as the file summary; later blocks are symbol documentation.
2. Determine the block's first tag line. The generator cuts the summary at the first line that starts with `@<tag>`. If that line is the block's first line, the rendered description is empty.
3. Confirm the rendered symptom in the generated `src/**/README.md`: the file heading `## <path>` must be followed by prose before the first symbol heading (`### <symbol>`). An immediate symbol heading means an empty summary.
4. If the block opens with a tag, rewrite it so the prose comes first and the module-identity tag moves to the **end** of the block. Preserve the existing prose verbatim — this skill changes ordering, not content.
5. Confirm the block contains **no** `@internal` tag. `@internal` in the leading block suppresses the entire file summary, not just the tag tail.
6. Keep the summary atemporal and self-contained: state what the module is and what a reader can do with it. Do not introduce plan language, phase numbers, or roadmap framing (Atemporal Public Docs Rule).
7. Run `npm run docs` to regenerate `src/**/README.md`, then re-read the affected file section and confirm the prose now renders before the first symbol heading.
8. Report the files changed, the before/after tag position, and the regenerated-output decision.

## Why the Ordering Rule Exists

The generated README pipeline reads the file summary through two functions in `scripts/generate-docs/symbols/generate-docs.symbols.jsdoc.utils.ts`:

1. `extractLeadingFileJsDocSummary` — takes the **first** `/** ... */` block at the top of the file.
2. `parseFileSummaryBlock` — computes `tagStartIndex = cleanedText.search(/^@\w+/m)` and keeps only the text **before** that index as the description.

Everything from the first tag onward is discarded from the summary (`@example` blocks are the only exception; they are re-extracted separately). Consequences:

- A block that opens with `@module neat.nge-collective.team-fitness` yields an **empty** description, and the generated README renders the file heading with no prose.
- A block that opens with prose and ends with `@module telemetry` yields the full prose summary.
- A leading block containing `@internal` yields **no** file summary at all — the generator returns early for the whole block.

This is a silent failure: the source looks documented, the validator passes, and only the rendered README is empty. The `docs-quality` scanner does **not** catch it — `scan_code_quality` inspects exported *symbols* for missing/weak JSDoc and cyclomatic complexity, never file-level leading blocks.

## Before / After Examples

**Before (summary renders empty):**

```ts
/**
 * @module neat.nge-collective.team-fitness
 *
 * Reusable team-level fitness evaluator seam for NGE collective benchmarks.
 */
```

**After (summary renders):**

```ts
/**
 * Reusable team-level fitness evaluator seam for NGE collective benchmarks.
 *
 * This module provides the core orchestration that maps generic team groups into
 * reusable team-fitness results using an injected aggregation policy.
 *
 * @module neat.nge-collective.team-fitness
 */
```

**Before (module identity tag first):**

```ts
/**
 * @module telemetry.policy
 * Policy helpers for the observability telemetry boundary.
 */
```

**After (prose first, tag last):**

```ts
/**
 * Policy helpers for the observability telemetry boundary.
 *
 * @module telemetry.policy
 */
```

## Decision Tree

```text
Flowchart summary: "Leading JSDoc block" → "First line a tag?"; "First line a tag?" → "Move tag to end" (Yes), "Check tag order" (No); "Move tag to end" → "Regenerate README"; "Check tag order" → "Tag before prose?"; "Tag before prose?" → "Reorder" (Yes), "Compliant" (No); "Reorder" → "Regenerate README"; "Compliant" → "Regenerate README"; "Regenerate README" → "Prose renders?"; "Prose renders?" → "Done" (Yes), "Still empty" (No); "Done"; "Still empty" → "Check @internal"; "Check @internal" → "Remove @internal" (Yes), "Escalate" (No); "Remove @internal" → "Regenerate README"; "Escalate".
```

## Guardrails

- Do not hand-edit generated `src/**/README.md` files; the fix is always the source JSDoc plus `npm run docs`.
- Do not reorder tags in **symbol-level** JSDoc blocks. The rule applies only to the leading file-level block; a symbol block may open with `@param`/`@returns` as usual.
- Do not delete prose to make a tag fit on the first line. Prose first is the requirement, not a formatting preference.
- Do not move `@example` blocks ahead of the prose — the first `@tag` line still truncates the summary, and `@example` blocks must stay after the prose to be rendered.
- Do not add `@internal` to the leading block of a public module.
- Do not introduce plan language, phase numbers, or PR references into the summary.
- Do not treat an empty README file section as a generator bug before checking the tag order of the source block.

## Expected Final Output

- The leading file-level JSDoc block of the target module opens with prose and carries its module-identity tag last.
- `npm run docs` regenerates `src/**/README.md`, and the affected file section shows prose before its first symbol heading.
- The before/after tag position, the regenerated-output decision, and any residual empty-summary modules found in the same pass are reported in the session output.
