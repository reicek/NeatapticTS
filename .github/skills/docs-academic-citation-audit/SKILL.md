---
name: docs-academic-citation-audit
description: 'Use when: auditing citations, attribution, or academic references in docs.'
argument-hint: 'Describe the documentation surface, source files, generated outputs, citations needed, and validation command.'
user-invocable: false
disable-model-invocation: false
skills:
  - educational-docs
  - license-attribution-audit
  - updating-js-docs
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Docs Academic Citation Audit

Use this skill when documentation changes need educational or academic quality,
particularly when named algorithms, external standards, or borrowed workflow
patterns are referenced without a citation.

This skill owns the standards for atemporal public documentation, Mermaid
diagram requirements, academic citation placement, and the validation pipeline
for generated README artifacts.

## When to Use

- A JSDoc block names an algorithm, formula, or technique without citing its
  academic origin.
- A folder-level README is missing a Mermaid diagram for an architecture
  overview, data flow, or state transition.
- An external reference was used to inform a workflow document but no citation
  exists in the skill or plan.
- A generated `src/**/README.md` was edited directly instead of via JSDoc and
  `npm run docs`.
- A documentation pass is using plan-language (roadmap references, PR numbers,
  phase markers) in a public-facing README that should be atemporal.
- A new algorithm is introduced in `src/` without a Wikipedia or paper citation
  in its JSDoc.

## When NOT to use

Do NOT use for license auditing - use `license-attribution-audit` instead. Do NOT use for generating READMEs - use `educational-docs` instead.

## Workflow Diagram

```text
Flowchart summary: "Scan docs for citations" → "Check each citation"; "Check each citation" → "Source exists?"; "Source exists?" → "Verify claim matches source" (Yes), "Flag missing citation" (No); "Verify claim matches source" → "Accurate?"; "Flag missing citation" → "Report findings"; "Accurate?" → "Pass" (Yes), "Flag inaccurate citation" (No); "Report findings"; "Pass"; "Flag inaccurate citation" → "Report findings".
```

## Task Packet

Pass a compact packet naming the documentation surface, the missing or
incorrect content, and the validation command.

```text
Use docs-academic-citation-audit for src/neat/selection/ JSDoc.
Surface: JSDoc in src/neat/selection/neat.selection.ts.
Issue: tournament selection described without a Wikipedia citation or paper reference.
Expected addition: Wikipedia link plus Stanley & Miikkulainen (2002) citation.
Validate with: npm run docs — confirm the generated README reflects the citation.
```

## Required Workflow

1. Keep public docs atemporal: concepts, invariants, tradeoffs, and examples
   instead of roadmap history, phase markers, or PR numbers.
2. Prefer source JSDoc over generated README edits for `src/**` documentation.
   Editing a generated `src/**/README.md` directly is a defect; fix the JSDoc
   and regenerate.
3. Add Mermaid diagrams when they teach faster than prose. Every folder-level
   chapter must have at least one diagram covering architecture, data flow,
   state transitions, or a decision flow.
4. Cite academic papers or external references when explaining named algorithms
   or borrowed workflow standards. Use Wikipedia as the primary public source;
   add the original paper (title and URL) when the algorithm has a canonical
   academic reference.
5. Credit sources proactively — never paraphrase a concept without attributing
   it. When touching existing documentation that contains un-attributed concepts,
   add the missing attribution rather than leaving the gap.
6. Keep examples short, dependency-light, and aligned to public APIs. Use
   fenced `ts` code blocks in JSDoc so the docs generator preserves them.
7. Run `npm run docs` when source JSDoc or generated example inputs change.
8. Verify generated output rather than patching generated files directly.

## Coordination

- Use `educational-docs` for broad documentation passes covering multiple files
  or subsystems.
- Use `license-attribution-audit` when external workflow sources inform
  customization docs and the license terms need to be checked.

## Before/After Citation Examples

**Before (weak citation):**

```md
NEAT uses a genetic algorithm to evolve networks.
```

**After (strong citation):**

```md
NEAT uses a genetic algorithm to evolve network topologies through
speciation and complexification [Stanley & Miikkulainen, 2002].

[Stanley & Miikkulainen, 2002]: K.O. Stanley and R. Miikkulainen,
"Evolving Neural Networks through Augmenting Topologies,"
Evolutionary Computation, vol. 10, no. 2, pp. 99-127, 2002.
```

## Guardrails

- Do not edit `src/**/README.md` files directly; they are generated artifacts.
  Fix the JSDoc source, then run `npm run docs`.
- Do not use plan-language (roadmap steps, phase names, PR numbers) in
  public-facing documentation; those belong in `plans/` only.
- Do not paraphrase a named algorithm without a citation; attribution is
  mandatory for every algorithm that has a formal origin.
- Do not omit Mermaid diagrams from folder-level chapters; a text-only README
  for a non-trivial module is incomplete.
- Do not skip `npm run docs` after JSDoc changes; unverified generated output
  is not a deliverable.

## Expected Final Output

A strong citation audit pass should produce:

- updated JSDoc with Wikipedia links and paper citations for every named
  algorithm touched,
- at least one Mermaid diagram added or updated for each folder-level chapter
  that was missing one,
- `npm run docs` output confirming the generated READMEs reflect the new
  citations,
- confirmation that no generated files were edited directly.
