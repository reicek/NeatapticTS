# Documentation Checklist For Splits

Use this checklist whenever a split touches exported or public code.

This checklist is intentionally narrow. For the full educational writing bar,
source-mapped README work, citations, or Wikimedia-safe visuals, invoke the
companion `educational-docs` skill.

## Required

- Every touched exported function, class, type, and shared constant has JSDoc.
- JSDoc explains what the API does and why the boundary exists.
- Parameters and returns are documented.
- Non-obvious semantics include a short example.
- Important invariants, defaults, error cases, or performance notes are called
  out.

## README-Aware Review

Ask these questions while looking at the generated folder README:

1. Would a first-time reader understand the purpose of the new subfolder?
2. Does the public facade description explain why it still exists?
3. Are helper files described in a way that sounds educational rather than
   merely mechanical?
4. Are any exported constants under-documented in the README because their
   source JSDoc is too terse?
5. Does the README make the module feel easier to explore after the split?

## Educational Quality Bar

- Prefer short conceptual descriptions over bare type restatement.
- Use brief examples that match the current public API.
- Hand the pass off to `educational-docs` when the docs need more than a local
  split-aware cleanup.
- Hand the pass off to `educational-docs` when the documentation also needs the
  shared Astro Bird visual language for Mermaid diagrams, tables, or callouts.
- Keep examples and explanations dependency-light so they survive doc
  generation cleanly.
- Avoid filler comments that say only what the code already states.